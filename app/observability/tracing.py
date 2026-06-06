from contextlib import contextmanager
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config.settings import settings


_initialized = False


def setup_tracing() -> None:
    global _initialized

    if _initialized:
        return

    if not settings.otel_enabled:
        _initialized = True
        return

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": "0.1.0",
        }
    )

    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_endpoint,
        insecure=True,
    )

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _initialized = True


def get_tracer():
    setup_tracing()
    return trace.get_tracer(settings.otel_service_name)


@contextmanager
def start_node_span(node_name: str, state: Optional[Dict[str, Any]] = None):
    tracer = get_tracer()

    state = state or {}

    with tracer.start_as_current_span(f"eventlens.node.{node_name}") as span:
        span.set_attribute("eventlens.node_name", node_name)

        if state.get("case_id"):
            span.set_attribute("eventlens.case_id", state.get("case_id"))

        if state.get("event_type"):
            span.set_attribute("eventlens.event_type", state.get("event_type"))

        if state.get("next_action"):
            span.set_attribute("eventlens.next_action", state.get("next_action"))

        span.set_attribute(
            "eventlens.recovery_attempts",
            int(state.get("recovery_attempts", 0) or 0),
        )

        span.set_attribute(
            "eventlens.llm_fallback_attempts",
            int(state.get("llm_fallback_attempts", 0) or 0),
        )

        yield span


def set_judge_attributes(span, state: Dict[str, Any]) -> None:
    judge_result = state.get("judge_result") or {}

    if not judge_result:
        return

    if judge_result.get("overall_status"):
        span.set_attribute("eventlens.judge_status", judge_result.get("overall_status"))

    if judge_result.get("judge_score") is not None:
        span.set_attribute("eventlens.judge_score", float(judge_result.get("judge_score")))

    if judge_result.get("should_recover") is not None:
        span.set_attribute(
            "eventlens.should_recover",
            bool(judge_result.get("should_recover")),
        )


def set_field_attributes(span, state: Dict[str, Any]) -> None:
    confidence_result = state.get("confidence_result") or {}
    scored_fields = confidence_result.get("scored_fields") or {}

    principal = scored_fields.get("principal_amount") or {}
    maturity = scored_fields.get("maturity_date") or {}

    if principal.get("value"):
        span.set_attribute("eventlens.principal_amount", str(principal.get("value")))

    if principal.get("final_confidence") is not None:
        span.set_attribute(
            "eventlens.principal_confidence",
            float(principal.get("final_confidence") or 0.0),
        )

    if maturity.get("value"):
        span.set_attribute("eventlens.maturity_date", str(maturity.get("value")))

    if maturity.get("final_confidence") is not None:
        span.set_attribute(
            "eventlens.maturity_confidence",
            float(maturity.get("final_confidence") or 0.0),
        )