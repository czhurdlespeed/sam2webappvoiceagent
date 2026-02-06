import asyncio
import json
import os
import time

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
    ChatContext,
    ChatMessage,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    function_tool,
    inference,
    metrics,
    room_io,
)
from livekit.plugins import lemonslice, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from parallel import AsyncParallel
from parallel.types.beta import ExcerptSettingsParam, SearchResult

from observability import setup_observability
from prompts import initial_assistant_prompt
from rag import RAG, MongoDB

# Load from project root so it works when cwd is not the repo root (e.g. in container job subprocess)
load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(
        self,
        parallel_client: AsyncParallel,
        rag: RAG,
        user_data: dict[str, str] | None = None,
    ) -> None:
        super().__init__(instructions=initial_assistant_prompt)
        assert isinstance(parallel_client, AsyncParallel), (
            "parallel_client must be an instance of AsyncParallel"
        )
        assert isinstance(rag, RAG), "rag must be an instance of RAG"
        self._parallel_client = parallel_client
        self._rag = rag
        self._user_data = user_data or {}

    async def on_enter(self):
        name = self._user_data.get("name", "").strip()
        if name:
            instructions = f"In one sentence, introduce yourself, welcome {name}, and ask how you can help them today."
        else:
            instructions = "In one sentence, introduce yourself, welcome the user, and ask them how you can help them today."
        await self.session.generate_reply(instructions=instructions).wait_for_playout()

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        try:
            rag_results = await self._rag.get_query_results(
                query=new_message.text_content,
                rag_top_k=15,
                rerank_top_k=3,
                verbose=False,
            )
            turn_ctx.add_message(
                role="assistant", content=RAG.format_results_for_llm(rag_results)
            )
        except Exception as e:
            logfire.error(
                f"RAG search failed for query: {new_message.text_content} with error: {e}"
            )
            turn_ctx.add_message(role="assistant", content=f"RAG search failed: {e}")

    @function_tool()
    async def web_search(
        self,
        context: RunContext,
        objective: str,
        search_queries: list[str],
    ) -> str:
        """
        Use this tool to search the internet if the user asks for further, more detailed information not contained within your chat context.

        Args:
            objective: Natural-language description of the web research goal, including source or freshness guidance and broader context from the task. Maximum 5000 characters.
            search_queries: Optional search queries to supplement the objective. Maximum 200 characters per query.

        Example:
            objective: ”I want to know when the UN was founded. Prefer UN’s websites.”
            search_queries: [“Founding year UN”, “Year of founding United Nations”]
        """
        context.disallow_interruptions()

        async def _speak_status_update(delay: float = 2):
            await asyncio.sleep(delay)
            await context.session.generate_reply(
                instructions="In one sentence, let the user know you are searching the web, it's taking a bit longer than expected, and will be right back!",
                allow_interruptions=False,
            ).wait_for_playout()

        speak_task = asyncio.create_task(_speak_status_update())

        try:
            websites: SearchResult = await self._parallel_client.beta.search(
                objective=objective,
                excerpts=ExcerptSettingsParam(max_chars_total=2_500),
                search_queries=search_queries,
                max_results=3,
                mode="agentic",
            )

            speak_task.cancel()
            await asyncio.gather(speak_task, return_exceptions=True)

            results = "Here are the top 3 results from the web search:\n\n"
            for search_result in websites.results:
                results += f"# {search_result.title}\n"
                results += f"URL: {search_result.url}\n"
                results += f"Published Date: {search_result.publish_date}\n"
                results += f"Excerpt: {search_result.excerpts}\n"
                results += f"{'':-^50}\n"
        except Exception as e:
            logfire.error("Web search failed", error=e)
            return f"Web search failed: {e}"

        return results


server = AgentServer()


def prewarm(proc: JobProcess):
    setup_observability()
    proc.userdata["parallel_client"] = AsyncParallel(
        api_key=os.getenv("PARALLEL_API_KEY")
    )
    mongodb = MongoDB()
    mongodb.connect()
    proc.userdata["rag"] = RAG(vector_database=mongodb)
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


async def on_session_end(ctx: JobContext) -> None:
    report = ctx.make_session_report()
    logfire.info(f"Session Report 📜👇\n{report}")


@server.rtc_session(on_session_end=on_session_end, agent_name="sam2webappassistant")
async def entrypoint(ctx: JobContext):
    user_data: dict[str, str] = {}
    if ctx.job.metadata:
        try:
            user_data = json.loads(ctx.job.metadata)
        except json.JSONDecodeError:
            pass
    logfire.info(f" Job metadata {ctx.job.metadata} parsed_user_data={user_data}")
    ctx.log_context_fields = {
        "room": ctx.room.name,
        **{k: v for k, v in user_data.items() if isinstance(v, str)},
    }

    session = AgentSession(
        stt=inference.STT(model="cartesia/ink-whisper", language="en"),
        llm=inference.LLM(
            model="google/gemini-2.5-flash",
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice=os.getenv("CARTESIA_VOICE_ID"),
            language="en",
            extra_kwargs={
                "emotion": "Enthusiastic",
                "pronunciation_dict_id": os.getenv("CARTESIA_PRONUNCIATION_DICT_ID"),
            },
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
    )

    async def session_time_limit():
        try:
            time_limit = int(os.getenv("SESSION_TIME_LIMIT_SECONDS", "90"))

            time_start = time.time()
            last_logged_interval = 0
            while time.time() - time_start < time_limit:
                await asyncio.sleep(1)
                elapsed = int(time.time() - time_start)
                current_interval = elapsed // 10
                if current_interval > last_logged_interval:
                    last_logged_interval = current_interval
                    logfire.info(
                        f"{time_limit - (time.time() - time_start):.0f} seconds remaining"
                    )
            logfire.info("Session time limit reached")
            # Generate the goodbye message
            await session.generate_reply(
                instructions="In one sentence, apologize and state that the session time limit has been reached and you must disconnect now.",
                allow_interruptions=False,
            ).wait_for_playout()

            # Disconnect the agent from the room
            await ctx.room.disconnect()
            await ctx.delete_room()

        except asyncio.CancelledError:
            pass  # Handle cleanup if the room ends early naturally

    timer_task = asyncio.create_task(session_time_limit())

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logfire.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.6),
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.6),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.5),
        ],
    )

    avatar = lemonslice.AvatarSession(
        agent_id="agent_682168d4ce7c95e0",
        agent_prompt="Enthusiastic and friendly assistant",
    )
    await avatar.start(session, room=ctx.room)

    await session.start(
        agent=Assistant(
            parallel_client=ctx.proc.userdata["parallel_client"],
            rag=ctx.proc.userdata["rag"],
            user_data=user_data,
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )
    await background_audio.start(room=ctx.room, agent_session=session)

    await ctx.connect()

    # Wait until the room disconnects (user left or timer fired)
    disconnect_fut = asyncio.get_running_loop().create_future()

    def _on_room_disconnected(*args):
        try:
            disconnect_fut.set_result(None)
        except asyncio.InvalidStateError:
            pass

    ctx.room.on("disconnected", _on_room_disconnected)
    await disconnect_fut

    timer_task.cancel()
    await asyncio.gather(timer_task, return_exceptions=True)


if __name__ == "__main__":
    cli.run_app(server)
