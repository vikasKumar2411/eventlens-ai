from functools import wraps
from typing import Any, Callable, Dict

from app.observability.tracing import (
    start_node_span,
    set_field_attributes,
    set_judge_attributes,
)


def traced_node(node_name: str):
    """
    Decorator for tracing EventLens graph nodes with OpenTelemetry.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(state: Dict[str, Any]):
            with start_node_span(node_name, state) as span:
                try:
                    result = func(state)

                    span.set_attribute("eventlens.node.success", True)

                    if isinstance(result, dict):
                        span.set_attribute(
                            "eventlens.node.output_keys",
                            ",".join(result.keys()),
                        )

                        merged_state = dict(state)
                        merged_state.update(result)

                        set_judge_attributes(span, merged_state)
                        set_field_attributes(span, merged_state)

                        if merged_state.get("next_action"):
                            span.set_attribute(
                                "eventlens.next_action",
                                merged_state.get("next_action"),
                            )

                        if merged_state.get("completed_steps"):
                            span.set_attribute(
                                "eventlens.completed_steps",
                                ",".join(merged_state.get("completed_steps", [])),
                            )

                    return result

                except Exception as exc:
                    span.set_attribute("eventlens.node.success", False)
                    span.set_attribute("eventlens.node.error", str(exc))
                    raise

        return wrapper

    return decorator