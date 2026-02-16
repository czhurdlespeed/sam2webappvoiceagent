import logging
import os

import logfire
from dotenv import load_dotenv
from livekit.agents.telemetry import set_tracer_provider

load_dotenv(".env.local")

# Guard to prevent duplicate setup when prewarm runs multiple times in the same process
_observability_initialized = False


def setup_observability():
    """Configure OpenTelemetry traces, metrics, and logs to export to Logfire."""
    global _observability_initialized

    if _observability_initialized:
        return
    _observability_initialized = True

    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if not logfire_token:
        raise ValueError(
            "LOGFIRE_TOKEN is not set. Add LOGFIRE_TOKEN to .env.local or set the environment variable."
        )

    def scrubbing_callback(m: logfire.ScrubMatch):
        if m.pattern_match.group(0) == "session":
            return m.value

    logfire.configure(
        scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback),
        service_name="voice-agent",
        token=logfire_token,
    )

    # Ensure LiveKit uses the same tracer provider that Logfire configured
    from opentelemetry import trace

    set_tracer_provider(trace.get_tracer_provider())

    # Optionally set log level for specific noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.INFO)

    logfire.info("OpenTelemetry observability setup complete (traces, metrics, logs)")
