import asyncio
import os
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import certifi
import pymongo
import regex as re
import voyageai
from dotenv import load_dotenv
from landingai_ade import LandingAIADE
from pymongo import AsyncMongoClient
from pymongo.operations import SearchIndexModel
from rich import print

load_dotenv(".env.local")

MONGO_URI = os.getenv("MONGO_URI")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_EMBEDDING_MODEL = os.getenv("VOYAGE_EMBEDDING_MODEL")
if not MONGO_URI:
    raise ValueError(
        "MONGO_URI is not set. Add MONGO_URI to .env.local or set the environment variable."
    )
if not VOYAGE_API_KEY:
    raise ValueError(
        "VOYAGE_API_KEY is not set. Add VOYAGE_API_KEY to .env.local or set the environment variable."
    )
if not VOYAGE_EMBEDDING_MODEL:
    raise ValueError(
        "VOYAGE_DOCUMENT_EMBEDDING_MODEL is not set. Add VOYAGE_DOCUMENT_EMBEDDING_MODEL to .env.local or set the environment variable."
    )


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
        self._client: AsyncMongoClient | None = None
        self._voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    @property
    def database(self) -> AsyncMongoClient | None:
        return self._client

    def connect(self) -> None:
        try:
            self._client = AsyncMongoClient(
                MONGO_URI,
                tlsCAFile=certifi.where(),
                server_api=pymongo.server_api.ServerApi(
                    version="1",
                    strict=False,
                    deprecation_errors=True,
                ),
            )
            print("[green]Connected successfully to MongoDB[/green]")
        except Exception as e:
            raise Exception(f"[red]Error connecting to MongoDB: {e}[/red]")

    async def disconnect(self) -> None:
        """Stop using the connection for this instance. Does not close the shared client."""
        if self._client is None:
            return
        await self._client.close()
        self._client = None

    async def create_collection(self, collection_name: str) -> None:
        if self._database is None:
            raise RuntimeError(
                "Not connected. Use await MongoDB.create() or await db.connect() first."
            )
        await self._database["sam2webappdocs"].create_collection(collection_name)
        print(f"[green]Collection {collection_name} created successfully[/green]")

    async def delete_collection(self, collection_name: str) -> None:
        if self._database is None:
            raise RuntimeError(
                "Not connected. Use await MongoDB.create() or await db.connect() first."
            )

        if collection_name not in await self.list_collections():
            print(f"[red]Collection {collection_name} not found in database[/red]")
            return
        await self._database["sam2webappdocs"].drop_collection(collection_name)
        print(f"[green]Collection {collection_name} deleted successfully[/green]")

    async def list_collections(self) -> List[str]:
        if self.database is None:
            raise RuntimeError(
                "Not connected. Use await MongoDB.create() or await db.connect() first."
            )
        return await self.database["sam2webappdocs"].list_collection_names()

    @property
    def embedding_model(self) -> Callable[List[List[str]], List[float]]:
        return partial(
            self._voyage_client.contextualized_embed,
            model=VOYAGE_EMBEDDING_MODEL,
            output_dimension=1024,
        )

    async def insert_embeddings(
        self, text_embedding_pairs: Iterator[Tuple[str, List[float]]]
    ) -> None:
        collection = self.database["sam2webappdocs"]["vectorstore"]
        docs_to_insert = []
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
        collection = self.database["sam2webappdocs"]["vectorstore"]
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
        def is_queryable(index: dict) -> bool:
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

    @staticmethod
    def _token_count(chunks: List[str]) -> int:
        client = voyageai.Client(api_key=VOYAGE_API_KEY)
        tokenized = client.tokenize(chunks, model="voyage-4-lite")
        return sum(len(t) for t in tokenized)

    @staticmethod
    def _split_document_if_too_many_tokens(
        chunks: List[str], document_name: str
    ) -> List[List[str]]:
        client = voyageai.Client(api_key=VOYAGE_API_KEY)
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

    async def upload_to_database(self, document_dir: Path) -> List:
        documents_chunks = self._chunk_documents(document_dir)
        batches = self._batch_by_tokens(documents_chunks)
        all_pairs: List[Tuple[List[str], object]] = []
        for batch in batches:
            results = self.vector_database.embedding_model(
                inputs=batch, input_type="document"
            ).results
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
    def format_results_for_llm(results: List[dict]) -> str:
        """Format RAG results as a concise string for LLM consumption."""
        if not results:
            return "No relevant documents found for this query."
        rag_string = "Here are the most relevant documents in the knowledge base for your query. The relevance score is a measure of how closely the document is related to the user's query. The documents are sorted by relevance score in descending order. Summarize the documents into a concise response that answers the user's question. The response will be used by a text-to-speech engine to be read aloud to the user.\n\n"
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
    mongodb = MongoDB()
    mongodb.connect()
    rag = RAG(vector_database=mongodb)
    results = await rag.get_query_results(
        query="How does fine-tuning SAM 2 help with manufacturing video object segmentation?",
        verbose=False,
    )
    print(RAG.format_results_for_llm(results))
    await mongodb.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
