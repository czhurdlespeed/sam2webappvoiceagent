import asyncio
import os

import logfire
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
    inference,
    mcp,
    room_io,
)
from livekit.plugins import lemonslice, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from observability import setup_observability
from prompts import initial_assistant_prompt
from rag import RAG, MongoDB

# Load from project root so it works when cwd is not the repo root (e.g. in container job subprocess)
load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, rag: RAG | None = None) -> None:
        self._rag = rag
        super().__init__(instructions=initial_assistant_prompt)

    async def on_enter(self):
        await self.session.say(
            text="Hello, I'm Calvin, your SAM 2 Web Application Assistant. How can I help you today?"
        )

    @function_tool()
    async def rag(
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
        try:
            _, results = await asyncio.gather(
                context.session.generate_reply(
                    instructions="Let user know you are searching your knowledge base for the answer to their question and will be right back!"
                ),
                self._rag.get_query_results(
                    query=query,
                    rag_top_k=15,
                    rerank_top_k=5,
                    verbose=False,
                ),
            )
            return RAG.format_results_for_llm(results)
        except Exception as e:
            logfire.error("RAG search failed", error=e)
            return f"RAG search failed: {e}"


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Observability setup (traces, metrics, logs)
    setup_observability()
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize RAG for documentation search (shared MongoDB connection is reused)
    try:
        mongodb = await MongoDB.create()
        rag = RAG(vector_database=mongodb)
    except Exception as e:
        logfire.warning("RAG not available: %s. Documentation search disabled.", e)
        rag = None

    session = AgentSession(
        stt=inference.STT(model="cartesia/ink-whisper", language="en"),
        llm=inference.LLM(
            model="gemini-2.5-flash",
            extra_kwargs={"max_completion_tokens": 150, "max_tokens": 150},
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="c2fadb00-50ee-4a1a-843c-6ccfe18663e9",
            language="en",
            extra_kwargs={"emotion": "Enthusiastic"},
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        mcp_servers=[
            mcp.MCPServerHTTP(
                url="https://search-mcp.parallel.ai/mcp",
                headers={"Authorization": "Bearer " + os.getenv("PARALLEL_API_KEY")},
                client_session_timeout_seconds=15,
            )
        ],
    )
    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.6),
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.6),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.5),
        ],
    )

    avatar = lemonslice.AvatarSession(
        agent_id="agent_682168d4ce7c95e0",
        agent_prompt="Be expressive, inviting, and helpful. Make sure to smile",
    )
    await avatar.start(session, room=ctx.room)

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

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
