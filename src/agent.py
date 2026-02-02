import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    mcp,
    room_io,
)
from livekit.plugins import cartesia, google, lemonslice, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from rag import RAG, MongoDB

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

fh = logging.FileHandler("agent.log")
fh.setFormatter(fmt)
logger.addHandler(fh)

ch = logging.StreamHandler()  # still print to console
ch.setFormatter(fmt)
logger.addHandler(ch)
# Load from project root so it works when cwd is not the repo root (e.g. in container job subprocess)
load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, rag: RAG | None = None) -> None:
        self._rag = rag
        super().__init__(
            instructions="""
            You are a helpful voice AI assistant for a web application that allows users to select LoRA parameter configurations
            and submit SAM 2 training jobs. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge and web search capabilities.
            When users ask about SAM 2, fine-tuning, LoRA, video object segmentation, manufacturing, or related topics, use the search_sam2_docs tool
            to retrieve relevant documentation before answering.
            Web search capabilities are provided by the Parallel MCP server. Use the web search capabilities to answer questions that don't relate to the RAG knowledge base.
            If you call the Parallel MCP web search server, let the user know you are searching the web for the answer to their question and will be right back!
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor but your ultimate priority is to help the user with their questions and tasks.
            """,
        )

    async def on_enter(self):
        await self.session.say(
            text="Hello, I'm your SAM 2 Web Application Assistant. How can I help you today?"
        )

    @function_tool()
    async def search_sam2_docs(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """Search the SAM 2 documentation and related materials for information relevant to the user's question.
        Use this tool when users ask about SAM 2, fine-tuning, LoRA, video object segmentation, manufacturing,
        or other topics covered in the documentation. Synthesize the retrieved content into a helpful response.

        Args:
            query: The user's question or topic to search for in the documentation.
        """
        if self._rag is None:
            return "Documentation search is not available. Please try asking your question another way."
        await context.session.generate_reply(
            instructions="Let user know you are searching your knowledge base for the answer to their question and will be right back!"
        )
        try:
            results = await self._rag.get_query_results(
                query=query,
                rag_top_k=15,
                rerank_top_k=5,
                verbose=False,
            )
            return RAG.format_results_for_llm(results)
        except Exception as e:
            logger.exception("RAG search failed")
            return f"Could not search the documentation: {e}"


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize RAG for documentation search (shared MongoDB connection is reused)
    try:
        mongodb = await MongoDB.create()
        rag = RAG(vector_database=mongodb)
    except Exception as e:
        logger.warning("RAG not available: %s. Documentation search disabled.", e)
        rag = None

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=cartesia.STT(model="ink-whisper", language="en"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(model="gemini-2.5-flash"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=cartesia.TTS(
            model="sonic-3",
            voice="c2fadb00-50ee-4a1a-843c-6ccfe18663e9",
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        mcp_servers=[
            mcp.MCPServerHTTP(
                url="https://search-mcp.parallel.ai/mcp",
                headers={"Authorization": "Bearer " + os.getenv("PARALLEL_API_KEY")},
                timeout=15,
                client_session_timeout_seconds=15,
            )
        ],
    )
    background_audio = BackgroundAudioPlayer(
        # play office ambience sound looping in the background
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
        # play keyboard typing sound when the agent is thinking
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
        ],
    )

    avatar = lemonslice.AvatarSession(
        agent_id="agent_682168d4ce7c95e0",
        agent_prompt="Be expressive and helpful. Make sure to smile",
    )
    await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(rag=rag),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )
    await background_audio.start(room=ctx.room, agent_session=session)

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
