import asyncio
import os
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Tuple

import certifi
import regex as re
import voyageai
from dotenv import load_dotenv
from landingai_ade import LandingAIADE
from pymongo import AsyncMongoClient, server_api
from pymongo.errors import ConnectionFailure
from pymongo.operations import SearchIndexModel
from rich import print
from voyageai.object.contextualized_embeddings import ContextualizedEmbeddingsResult

load_dotenv(".env.local")


class VectorDatabase(ABC):
    @property
    @abstractmethod
    def database(self):
        pass

    @property
    @abstractmethod
    def embedding_model(self):
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass


def parse_pdfs(pdf_directory: Path) -> None:
    client = LandingAIADE()

    for pdf_file in (pdf_directory / "pdfs").glob("*.pdf"):
        print(f"[blue]Parsing PDF: {pdf_file.name}[/blue]")
        try:
            response = client.parse(document=pdf_file, model="dpt-2-latest")
        except Exception as e:
            print(f"[red]Error parsing PDF: {e}[/red]")
            continue
        with open(Path(f"{pdf_directory}/markdown/{pdf_file.stem}.md"), "w") as f:
            f.write(response.markdown)
            print(f"[green]Markdown file created: {pdf_file.stem}.md[/green]")
        print(f"[blue]PDF {pdf_file.name} parsed successfully[/blue]")


class MongoDB(VectorDatabase):
    def __init__(self) -> None:
        self.load_env()
        self._client: AsyncMongoClient | None = None

    def load_env(self) -> None:
        VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
        VOYAGE_EMBEDDING_MODEL = os.getenv("VOYAGE_EMBEDDING_MODEL")
        MONGO_URI = os.getenv("MONGO_URI")
        if not VOYAGE_API_KEY:
            raise ValueError(
                "VOYAGE_API_KEY is not set. Add VOYAGE_API_KEY to .env.local or set the environment variable."
            )
        if not VOYAGE_EMBEDDING_MODEL:
            raise ValueError(
                "VOYAGE_EMBEDDING_MODEL is not set. Add VOYAGE_EMBEDDING_MODEL to .env.local or set the environment variable."
            )
        if not MONGO_URI:
            raise ValueError(
                "MONGO_URI is not set. Add MONGO_URI to .env.local or set the environment variable."
            )
        self._voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        self._embedding_model = VOYAGE_EMBEDDING_MODEL
        self._mongo_uri = MONGO_URI

    @property
    def database(self) -> AsyncMongoClient:
        if self._client is None:
            raise RuntimeError(
                "Not connected. Use await MongoDB.create() or await db.connect() first."
            )
        return self._client

    @classmethod
    async def connect(cls) -> "MongoDB":
        try:
            instance = cls()
            instance._client = AsyncMongoClient(
                instance._mongo_uri,
                tlsCAFile=certifi.where(),
                server_api=server_api.ServerApi(
                    version="1",
                    strict=False,
                    deprecation_errors=True,
                ),
            )
            await instance._client.admin.command("ping")
            print("[green]Connected successfully to MongoDB[/green]")
            return instance
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Error connecting to MongoDB: {e}")

    async def disconnect(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.close()
        except Exception as e:
            raise Exception(f"Error disconnecting from MongoDB: {e}")

    async def create_collection(self, collection_name: str) -> None:
        if self._client is None:
            raise RuntimeError("Not connected. Use await MongoDB.connect() first.")
        assert isinstance(self._client, AsyncMongoClient)
        await self._client["sam2webappdocs"].create_collection(collection_name)
        print(f"[green]Collection {collection_name} created successfully[/green]")

    async def delete_collection(self, collection_name: str) -> None:
        if self._client is None:
            raise RuntimeError("Not connected. Use await MongoDB.connect() first.")

        if collection_name not in await self.list_collections():
            print(f"[red]Collection {collection_name} not found in database[/red]")
            return
        assert isinstance(self._client, AsyncMongoClient)
        await self._client["sam2webappdocs"].drop_collection(collection_name)
        print(f"[green]Collection {collection_name} deleted successfully[/green]")

    async def list_collections(self) -> list[str]:
        if self._client is None:
            raise RuntimeError("Not connected. Use await MongoDB.connect() first.")
        assert isinstance(self._client, AsyncMongoClient)
        return await self._client["sam2webappdocs"].list_collection_names()

    @property
    def embedding_model(self) -> Callable[..., Any]:
        return partial(
            self._voyage_client.contextualized_embed,
            model=self._embedding_model,
            output_dimension=1024,
        )

    async def insert_embeddings(
        self,
        text_embedding_pairs: List[Tuple[List[str], ContextualizedEmbeddingsResult]],
    ) -> None:
        if self._client is None:
            raise RuntimeError("Not connected. Use await MongoDB.connect() first.")
        assert isinstance(self._client, AsyncMongoClient)
        collection = self._client["sam2webappdocs"]["vectorstore"]
        docs_to_insert: List[Dict[str, str | float]] = []
        for document, document_embeddings in text_embedding_pairs:
            for text, embedding in zip(document, document_embeddings.embeddings):
                docs_to_insert.append(
                    {
                        "text": text,
                        "embedding": embedding,
                    }
                )
        await collection.insert_many(docs_to_insert)

    async def create_vector_index(self) -> None:
        assert isinstance(self._client, AsyncMongoClient)
        collection = self._client["sam2webappdocs"]["vectorstore"]
        index_name = "vector_index"
        cursor = await collection.list_search_indexes(index_name)
        existing_indexes = await cursor.to_list()
        if existing_indexes:
            if existing_indexes[0].get("queryable"):
                return
        else:
            # Create the search index
            print(f"[green]Creating vector search index {index_name}...[/green]")
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "numDimensions": 1024,
                            "path": "embedding",
                            "similarity": "dotProduct",
                        }
                    ]
                },
                name=index_name,
                type="vectorSearch",
            )
            await collection.create_search_index(model=search_index_model)
            print(
                f"[green]Vector search index {index_name} created successfully[/green]"
            )

        # Wait for index to become queryable
        def is_queryable(index: Mapping[str, Any]) -> bool:
            return index.get("queryable") is True

        while True:
            cursor = await collection.list_search_indexes(index_name)
            indices = await cursor.to_list()
            if len(indices) and is_queryable(indices[0]):
                break
            await asyncio.sleep(5)


class RAG:
    def __init__(self, vector_database: MongoDB):
        self.vector_database = vector_database

    BATCH_TOKEN_LIMIT = 120_000

    def _token_count(self, chunks: List[str]) -> int:
        client = self.vector_database._voyage_client
        tokenized = client.tokenize(chunks, model="voyage-4-lite")
        return sum(len(t) for t in tokenized)

    def _split_document_if_too_many_tokens(
        self, chunks: List[str], document_name: str
    ) -> List[List[str]]:
        client = self.vector_database._voyage_client
        tokenized = client.tokenize(chunks, model="voyage-4-lite")
        total_tokens = sum(len(chunk) for chunk in tokenized)
        if total_tokens <= 32_000:
            return [chunks]
        print(
            f"[yellow]Document {document_name} has {total_tokens} tokens, which is too many. Splitting into chunks...[/yellow]"
        )
        split_indexes = []
        index = 0
        token_count = 0
        while index < len(tokenized):
            token_count += len(tokenized[index])
            if token_count > 32_000:
                split_indexes.append(index)
                token_count = len(tokenized[index])
            index += 1
        boundaries = [0] + split_indexes + [len(chunks)]
        return [
            chunks[boundaries[i] : boundaries[i + 1]]
            for i in range(len(boundaries) - 1)
        ]

    def _chunk_documents(self, document_dir: Path) -> List[List[str]]:
        documents_chunks = []
        for document in document_dir.glob("*.md"):
            with open(document, "r") as f:
                text = f.read()
            chunks = re.split(
                r"<a id='[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'></a>",
                text,
            )
            chunks = list(filter(lambda x: x != "", chunks))
            documents_chunks.extend(
                self._split_document_if_too_many_tokens(chunks, document.name)
            )
        return documents_chunks

    def _batch_by_tokens(
        self, documents_chunks: List[List[str]]
    ) -> List[List[List[str]]]:
        batches: List[List[List[str]]] = []
        current_batch: List[List[str]] = []
        current_tokens = 0
        for doc in documents_chunks:
            doc_tokens = self._token_count(doc)
            if current_tokens + doc_tokens > self.BATCH_TOKEN_LIMIT and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(doc)
            current_tokens += doc_tokens
        if current_batch:
            batches.append(current_batch)
        return batches

    async def upload_to_database(self, document_dir: Path) -> None:
        documents_chunks = self._chunk_documents(document_dir)
        batches = self._batch_by_tokens(documents_chunks)
        all_pairs: List[Tuple[List[str], ContextualizedEmbeddingsResult]] = []
        for batch in batches:
            results: List[ContextualizedEmbeddingsResult] = (
                self.vector_database.embedding_model(
                    inputs=batch, input_type="document"
                ).results
            )
            for doc, doc_embeddings in zip(batch, results):
                all_pairs.append((doc, doc_embeddings))
        await self.vector_database.insert_embeddings(iter(all_pairs))

    async def get_query_results(
        self,
        query: str,
        rag_top_k: int = 15,
        rerank_top_k: int = 5,
        verbose: bool = True,
    ) -> List[dict]:
        collection = self.vector_database.database["sam2webappdocs"]["vectorstore"]
        query_embedding = (
            self.vector_database.embedding_model(inputs=[[query]], input_type="query")
            .results[0]
            .embeddings[0]
        )
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "exact": True,
                    "limit": rag_top_k,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        results = await collection.aggregate(pipeline)
        results = await results.to_list()
        if not results:
            return []
        documents = [doc["text"] for doc in results]
        reranking = self.vector_database._voyage_client.rerank(
            query=query,
            documents=documents,
            model="rerank-2.5",
            top_k=min(rerank_top_k, len(documents)),
        )
        reranked = [
            {"text": r.document, "index": r.index, "relevance_score": r.relevance_score}
            for r in reranking.results
        ]
        if verbose:
            for doc in reranked:
                print(doc)
        return reranked

    @staticmethod
    def format_results_for_llm(
        results: List[dict], relavence_threshold: float = 0.8
    ) -> str:
        """Format RAG results as a concise string for LLM consumption."""
        results = list(
            filter(lambda x: x.get("relevance_score") >= relavence_threshold, results)
        )
        if not results:
            return ""
        rag_string = "Additional information from the knowledge base relevant to the user's query:\n\n"
        for idx, doc in enumerate(results, 1):
            string = ""
            text = doc.get("text", "").strip()
            score = doc.get("relevance_score")
            string += f"# Document [{idx}]\n"
            string += (
                f"# Relevance Score: {score:.2f}\n"
                if isinstance(score, float)
                else "Relevance Score: N/A"
            )
            string += "# Document Content:\n\n"
            string += text + "\n\n"
            string += f"{'':-^50}\n"
            rag_string += string
        return rag_string


async def main() -> None:
    mongodb = await MongoDB.connect()

    rag = RAG(vector_database=mongodb)
    results = await rag.get_query_results(
        query="How does fine-tuning SAM 2 help with manufacturing video object segmentation?",
        verbose=False,
    )
    print(RAG.format_results_for_llm(results, relavence_threshold=0.75))
    await mongodb.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
