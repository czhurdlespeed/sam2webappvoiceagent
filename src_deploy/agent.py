import asyncio
import os
import time

import logfire
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
    JobRequest,
    MetricsCollectedEvent,
    RunContext,
    cli,
    function_tool,
    metrics,
    room_io,
)
from livekit.api import DeleteRoomRequest, LiveKitAPI
from livekit.plugins import (
    cartesia,
    deepgram,
    lemonslice,
    noise_cancellation,
    openai,
    silero,
)
from parallel import AsyncParallel
from parallel.types.beta import ExcerptSettingsParam, SearchResult
from pydantic import ValidationError
from pydantic_core import from_json
from pymongo.errors import PyMongoError

from .email import (
    confirm_conversation_history,
    generate_summary,
    send_email,
    template_email,
)
from .observability import setup_observability
from .prompts import initial_assistant_prompt
from .rag import RAG, MongoDB
from .usersession import (
    EmbedUserData,
    UserAgentUsage,
    UserSession,
    VoiceAgentUsageDatabase,
)


class Assistant(Agent):
    def __init__(
        self,
        rag: RAG,
        user_data: dict[str, str] | None = None,
    ) -> None:
        super().__init__(instructions=initial_assistant_prompt)
        assert isinstance(rag, RAG), "rag must be an instance of RAG"
        self._rag = rag
        self._user_data = user_data or {}
        self._parallel_client = AsyncParallel(api_key=os.getenv("PARALLEL_API_KEY"))

    async def on_enter(self):
        name = self._user_data.get("name", "").strip().split(" ")[0]
        if name:
            instructions = f"In one sentence, welcome {name},  introduce yourself, and ask how you can help them today."
        else:
            instructions = "In one sentence, welcome the user, introduce yourself, and ask them how you can help them today."
        await self.session.generate_reply(
            instructions=instructions, allow_interruptions=False
        ).wait_for_playout()

    @function_tool()
    async def search_knowledge_base(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """
        Use this tool to search the RAG knowledge base for information about low rank adaptation, configurations, manufacturing processes, or training AI models.
        """

        async def _speak_status_update(delay: float = 2):
            await asyncio.sleep(delay)
            await context.session.generate_reply(
                instructions="In one sentence, let the user know you are searching the knowledge base, it's taking a bit longer than expected, and will be right back!",
                allow_interruptions=False,
            )

        speak_task = asyncio.create_task(_speak_status_update())
        try:
            rag_results = await self._rag.get_query_results(
                query=query,
                rag_top_k=15,
                rerank_top_k=5,
                verbose=False,
            )
            return RAG.format_results_for_llm(rag_results, relavence_threshold=0.75)
        except asyncio.CancelledError:
            raise
        except PyMongoError as e:
            logfire.error("RAG search failed", error=e)
            return "Sorry, I couldn’t access the knowledge base just now. Please try again."
        finally:
            speak_task.cancel()
            await asyncio.gather(speak_task, return_exceptions=True)

    @function_tool()
    async def web_search(
        self,
        context: RunContext,
        objective: str,
        search_queries: list[str],
    ) -> str:
        """
        Use this tool to search the internet if the user asks for information that was not found in the RAG knowledge base.

        Args:
            objective: Natural-language description of the web research goal, including source or freshness guidance and broader context from the task. Maximum 5000 characters.
            search_queries: Optional search queries to supplement the objective. Maximum 200 characters per query.

        Example:
            objective: ”I want to know when the UN was founded. Prefer UN’s websites.”
            search_queries: [“Founding year UN”, “Year of founding United Nations”]
        """

        async def _speak_status_update(delay: float = 2):
            await asyncio.sleep(delay)
            await context.session.generate_reply(
                instructions="In one sentence, let the user know you are searching the web, it's taking a bit longer than expected, and will be right back!",
                allow_interruptions=False,
            )

        speak_task = asyncio.create_task(_speak_status_update())
        try:
            websites: SearchResult = await self._parallel_client.beta.search(
                objective=objective,
                excerpts=ExcerptSettingsParam(max_chars_total=2_500),
                search_queries=search_queries,
                max_results=3,
                mode="agentic",
            )

            results = "Here are the top 3 results from the web search:\n\n"
            for search_result in websites.results:
                results += f"# {search_result.title}\n"
                results += f"URL: {search_result.url}\n"
                results += f"Published Date: {search_result.publish_date}\n"
                results += f"Excerpt: {search_result.excerpts}\n"
                results += f"{'':-^50}\n"
            return results
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logfire.error("Web search failed", error=e)
            return "Web search failed."
        finally:
            speak_task.cancel()
            await asyncio.gather(speak_task, return_exceptions=True)


server = AgentServer()
try:
    setup_observability()
except Exception:
    pass


def prewarm(proc: JobProcess):
    setup_observability()
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


async def on_session_end(ctx: JobContext) -> None:
    try:
        report = ctx.make_session_report()
        logfire.info(f"Session Report 📜👇\n{report}")
    except Exception as e:
        logfire.error("Failed to make session report", error=e)


def parse_metadata(metadata: str) -> EmbedUserData | None:
    try:
        return EmbedUserData.model_validate(from_json(metadata, allow_partial=True))
    except ValidationError as e:
        logfire.error("Failed to parse metadata", error=e, exc_info=True)
        return None


async def on_request(request: JobRequest) -> None:
    """Reject jobs that don't have valid user_id in metadata before the job runs."""
    logfire.info(f"Metadata: {request.job.metadata}")
    if not request.job.metadata or request.job.metadata.strip() == "":
        logfire.error("Job rejected: no metadata provided", job_id=request.id)
        async with LiveKitAPI() as lkapi:
            await lkapi.room.delete_room(DeleteRoomRequest(room=request.room.name))
        await request.reject()
        return
    user_data = parse_metadata(request.job.metadata)
    if user_data is None or not user_data.user_id or not user_data.user_id.strip():
        logfire.error(
            "Job rejected: invalid or missing user_id in metadata",
            job_id=request.id,
            metadata=request.job.metadata,
        )
        async with LiveKitAPI() as lkapi:
            await lkapi.room.delete_room(DeleteRoomRequest(room=request.room.name))
        await request.reject()
        return
    async with VoiceAgentUsageDatabase() as database:
        user_exists_in_db = await database.user_exists(user_data.user_id)
        if not user_exists_in_db:
            logfire.error(
                "Job rejected: user does not exist",
                job_id=request.id,
                user_id=user_data.user_id,
            )
            async with LiveKitAPI() as lkapi:
                await lkapi.room.delete_room(DeleteRoomRequest(room=request.room.name))
            await request.reject()
            return
        time_used = await database.fetch_user_time_used(user_data.user_id)
        if time_used >= int(os.getenv("SESSION_TIME_LIMIT_SECONDS", "120")):
            logfire.error(
                "Job rejected: user has reached the session time limit",
                job_id=request.id,
                user_id=user_data.user_id,
            )
            async with LiveKitAPI() as lkapi:
                await lkapi.room.delete_room(DeleteRoomRequest(room=request.room.name))
            await request.reject()
            return
        await request.accept()


@server.rtc_session(
    on_request=on_request,
    on_session_end=on_session_end,
    agent_name="sam2webappassistant",
)
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    user_data = parse_metadata(ctx.job.metadata)
    assert user_data is not None, "User id metadata is required"
    ctx.log_context_fields.update(**user_data.model_dump())
    try:
        async with VoiceAgentUsageDatabase() as database:
            user_agent_usage: UserAgentUsage = (
                await database.fetch_user_info_and_sessions(
                    user_data.user_id,
                )
            )

        logfire.info("User info", user_agent_usage=user_agent_usage.model_dump())
        if (num_sessions := len(user_agent_usage.session_ids)) == 0:
            logfire.info(
                f"User {user_data.user_id} has no user sessions, creating first user session",
                num_sessions=num_sessions,
            )
            user_session: UserSession = UserSession(
                user_id=user_data.user_id,
                session_id=num_sessions + 1,
                seconds_used=0,
            )
            logfire.info("User session created", info=user_session.model_dump())
        else:
            logfire.info(
                f"User has {num_sessions} user sessions, updating with existing user sessions"
            )
            user_session: UserSession = UserSession(
                user_id=user_data.user_id,
                session_id=num_sessions + 1,
                seconds_used=sum(user_agent_usage.seconds_used),
            )
            logfire.info(
                f"User has used {user_session.seconds_used} seconds in previous sessions."
            )
    except Exception as e:
        logfire.error(
            "Failed to acquire or set user metadata, info, and/or agent usage",
            error=e,
            exc_info=True,
        )
        try:
            await ctx.room.disconnect()
            await ctx.delete_room(room_name=ctx.room.name)
            await ctx.api.aclose()
        except Exception as e:
            logfire.error("Failed to disconnect or delete room", error=e, exc_info=True)
        finally:
            return

    session = AgentSession(
        stt=deepgram.STTv2(
            model="flux-general-en",
            eager_eot_threshold=0.7,
            eot_threshold=0.7,
            eot_timeout_ms=1_000,
        ),
        llm=openai.responses.LLM(
            model="gpt-4.1-nano",
            parallel_tool_calls=False,
        ),
        tts=cartesia.TTS(
            model="sonic-3",
            voice=str(os.getenv("CARTESIA_VOICE_ID")),
            emotion="Enthusiastic",
            pronunciation_dict_id=str(os.getenv("CARTESIA_PRONUNCIATION_DICT_ID")),
        ),
        turn_detection="stt",
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
        max_tool_steps=1,
    )

    time_start = time.time()

    session_ready = asyncio.Event()
    time_limit_handled = False  # Flag to track if time limit cleanup was already done

    async def session_time_limit(user_session: UserSession):
        logfire.info(
            "Timer started",
            time_limit_env=os.getenv("SESSION_TIME_LIMIT_SECONDS", "120"),
            has_user_session=user_session is not None,
        )
        try:
            time_limit = int(os.getenv("SESSION_TIME_LIMIT_SECONDS", "120")) - (
                user_session.seconds_used if user_session else 0
            )

            elapsed = 0
            if time_limit > 0:
                last_logged_interval = 0
                while time.time() - time_start < time_limit:
                    await asyncio.sleep(1)
                    elapsed = int(time.time() - time_start)
                    current_interval = elapsed // 10
                    if current_interval > last_logged_interval:
                        last_logged_interval = current_interval
                        logfire.info(f"{time_limit - elapsed:.0f} seconds remaining")
            else:
                elapsed = int(time.time() - time_start)
                await session_ready.wait()

            logfire.info(
                "Session time limit reached", time_limit=time_limit, elapsed=elapsed
            )
            try:
                await session.interrupt(force=True)
                if user_session.session_id > 0:
                    await session.say(
                        text="I'm sorry, but your sessions have reached the time limit and I must disconnect now. Goodbye!",
                        allow_interruptions=False,
                    ).wait_for_playout()
                else:
                    await session.generate_reply(
                        instructions="I'm sorry, but this session has reached the time limit and I must disconnect now. Goodbye!",
                        allow_interruptions=False,
                    ).wait_for_playout()

                # Write session to database and send email before deleting room
                seconds_used = int(time.time() - time_start)
                user_session.seconds_used = seconds_used
                logfire.info(
                    f"Writing session to database (time limit). Seconds used: {seconds_used}"
                )
                try:
                    async with VoiceAgentUsageDatabase() as database:
                        await asyncio.wait_for(
                            asyncio.shield(database.write_user_session(user_session)),
                            timeout=5,
                        )
                        logfire.info(
                            "User session written to database (time limit)",
                            user_agent_usage=user_session.model_dump(),
                        )
                except Exception as e:
                    logfire.error(
                        "Failed to write user session to database (time limit)",
                        error=e,
                        exc_info=True,
                    )

                # Send email if conversation history exists and email is verified
                has_conversation_history = confirm_conversation_history(
                    session.history.to_provider_format(format="openai")[0]
                )
                email_verified = user_agent_usage.email_verified

                logfire.info(
                    "Email sending check (time limit)",
                    has_conversation_history=has_conversation_history,
                    email_verified=email_verified,
                    will_send_email=has_conversation_history and email_verified,
                )

                if has_conversation_history and email_verified:

                    async def send_email_workflow():
                        summary = await generate_summary(
                            session.history.to_provider_format(format="openai")[0]
                        )
                        logfire.info("Summary generated (time limit)", summary=summary)
                        total_seconds = (
                            sum(user_agent_usage.seconds_used)
                            + user_session.seconds_used
                        )
                        session_count = len(user_agent_usage.session_ids) + 1
                        html = template_email(
                            user_agent_usage,
                            summary,
                            total_seconds_used=total_seconds,
                            session_count=session_count,
                        )
                        logfire.info("HTML generated (time limit)", html=html)
                        send_email(
                            user_agent_usage,
                            html,
                            session_count=session_count,
                        )

                    try:
                        await asyncio.wait_for(
                            asyncio.shield(send_email_workflow()),
                            timeout=15,
                        )
                        logfire.info("Email sent successfully after session time limit")
                    except Exception as e:
                        logfire.error(
                            "Failed to send email (time limit)", error=e, exc_info=True
                        )
                else:
                    logfire.info(
                        "Email not sent (time limit)",
                        reason="missing_conversation_history"
                        if not has_conversation_history
                        else "email_not_verified",
                    )

            except Exception as e:
                logfire.error(
                    "Session time limit: failed to announce or disconnect",
                    error=e,
                    exc_info=True,
                )
            finally:
                nonlocal time_limit_handled
                time_limit_handled = True
                _complete_disconnect()

        except asyncio.CancelledError:
            logfire.info("Timer cancelled")

    timer_task = asyncio.create_task(session_time_limit(user_session))
    logfire.info(
        "Timer task created",
        task_pending=not timer_task.done(),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logfire.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.5),
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.5),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.4),
        ],
    )

    avatar = lemonslice.AvatarSession(
        agent_id="agent_682168d4ce7c95e0",
        agent_prompt="Enthusiastic and friendly assistant",
    )

    mongodb = await MongoDB.connect()
    rag = RAG(vector_database=mongodb)

    # Wait until the room disconnects (user left or timer fired)
    disconnect_fut = asyncio.get_running_loop().create_future()

    def _complete_disconnect():
        try:
            disconnect_fut.set_result(None)
        except asyncio.InvalidStateError:
            pass

    def _on_room_disconnected(*args):
        _complete_disconnect()

    def _on_participant_disconnected(
        participant: rtc.RemoteParticipant,
        *_args,
    ):
        logfire.info(
            "Participant disconnected, shutting down",
            disconnected_identity=participant.identity,
        )
        asyncio.create_task(_disconnect_room())

    async def _disconnect_room():
        await mongodb.disconnect()
        _complete_disconnect()

    ctx.room.on("disconnected", _on_room_disconnected)
    ctx.room.on("participant_disconnected", _on_participant_disconnected)

    try:
        await avatar.start(session, room=ctx.room)

        await session.start(
            agent=Assistant(
                rag=rag,
                user_data={
                    "name": user_agent_usage.name,
                    "email": user_agent_usage.email,
                },
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
        session_ready.set()
        await disconnect_fut
    finally:
        timer_task.cancel()
        try:
            await asyncio.wait_for(timer_task, timeout=2)
        except asyncio.CancelledError:
            pass  # Expected when we cancel the task
        except asyncio.TimeoutError:
            pass  # Timer may be in generate_reply during shutdown—proceed with cleanup
        except BaseException:
            pass  # Session/room may be closing; any exception here is expected during cleanup

        # Do DB write and email before delete_room(); the worker may shut down once the room is deleted
        # Skip if time limit already handled these operations
        if user_session and not time_limit_handled:
            seconds_used = int(time.time() - time_start)
            user_session.seconds_used = seconds_used
            logfire.info(f"Writing session to database. Seconds used: {seconds_used}")
            try:
                async with VoiceAgentUsageDatabase() as database:
                    await asyncio.wait_for(
                        asyncio.shield(database.write_user_session(user_session)),
                        timeout=5,
                    )
                    logfire.info(
                        "User session written to database",
                        user_agent_usage=user_session.model_dump(),
                    )
            except Exception as e:
                logfire.error(
                    "Failed to write user session to database", error=e, exc_info=True
                )
            # Check if email should be sent (conversation history exists and email is verified)
            has_conversation_history = confirm_conversation_history(
                session.history.to_provider_format(format="openai")[0]
            )
            email_verified = user_agent_usage.email_verified

            logfire.info(
                "Email sending check",
                has_conversation_history=has_conversation_history,
                email_verified=email_verified,
                will_send_email=has_conversation_history and email_verified,
            )

            if has_conversation_history and email_verified:

                async def send_email_workflow():
                    summary = await generate_summary(
                        session.history.to_provider_format(format="openai")[0]
                    )
                    logfire.info("Summary generated", summary=summary)
                    total_seconds = (
                        sum(user_agent_usage.seconds_used) + user_session.seconds_used
                    )
                    session_count = len(user_agent_usage.session_ids) + 1
                    html = template_email(
                        user_agent_usage,
                        summary,
                        total_seconds_used=total_seconds,
                        session_count=session_count,
                    )
                    logfire.info("HTML generated", html=html)
                    send_email(
                        user_agent_usage,
                        html,
                        session_count=session_count,
                    )

                try:
                    await asyncio.wait_for(
                        asyncio.shield(send_email_workflow()),
                        timeout=15,
                    )
                    logfire.info("Email sent successfully after session end")
                except Exception as e:
                    logfire.error("Failed to send email", error=e, exc_info=True)
            else:
                logfire.info(
                    "Email not sent",
                    reason="missing_conversation_history"
                    if not has_conversation_history
                    else "email_not_verified",
                )

        try:
            await ctx.room.disconnect()
            await ctx.delete_room(room_name=ctx.room.name)
            await session.aclose()
        except Exception as e:
            logfire.warning("Failed to shutdown the session", error=e)


if __name__ == "__main__":
    cli.run_app(server)
