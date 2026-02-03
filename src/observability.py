import logging
import os

import logfire
from dotenv import load_dotenv
from livekit.agents.telemetry import set_tracer_provider

load_dotenv(".env.local")


def setup_observability():
    """Configure OpenTelemetry traces, metrics, and logs to export to Logfire."""

    # Get configuration from environment
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    otlp_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")

    # Parse headers string into dict (format: "Key1=Value1,Key2=Value2")
    headers = {}
    if otlp_headers:
        for pair in otlp_headers.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                headers[key.strip()] = value.strip()

    # Configure Logfire (provides its own instrumentation)
    logfire.configure(service_name="voice-agent", token=logfire_token)

    # Import OpenTelemetry components
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # Create a shared resource for all telemetry
    resource = Resource.create(attributes={SERVICE_NAME: "voice-agent"})

    # --- Traces ---
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{otlp_endpoint}/v1/traces",
                headers=headers,
            )
        )
    )
    set_tracer_provider(trace_provider)  # LiveKit-specific

    # --- Logs ---
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(
                endpoint=f"{otlp_endpoint}/v1/logs",
                headers=headers,
            )
        )
    )

    # Attach OTEL handler to root logger to capture all Python logging
    otel_handler = LoggingHandler(
        level=logging.DEBUG,
        logger_provider=logger_provider,
    )
    logging.getLogger().addHandler(otel_handler)

    # Optionally set log level for specific noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.INFO)

    logfire.info("OpenTelemetry observability setup complete (traces, metrics, logs)")
