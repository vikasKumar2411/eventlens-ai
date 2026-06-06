import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from app.config.settings import settings
from app.observability.tracing import get_tracer


class LLMService:
    """
    Thin wrapper around local Ollama.

    Future LLM agents should call this service instead of calling Ollama directly.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[int] = None,
    ):
        self.model = model or settings.llm_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.temperature = (
            temperature
            if temperature is not None
            else settings.llm_temperature
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.llm_timeout_seconds
        )

    def generate_text(self, prompt: str) -> str:
        tracer = get_tracer()

        with tracer.start_as_current_span("eventlens.service.llm.generate_text") as span:
            span.set_attribute("llm.provider", "ollama")
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.base_url", self.base_url)
            span.set_attribute("llm.temperature", float(self.temperature))
            span.set_attribute("llm.prompt_chars", len(prompt))

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                },
            }

            response = self._post_ollama(
                endpoint="/api/generate",
                payload=payload,
            )

            text = response.get("response", "").strip()

            span.set_attribute("llm.response_chars", len(text))
            span.set_attribute("llm.success", True)

            return text

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        tracer = get_tracer()

        with tracer.start_as_current_span("eventlens.service.llm.generate_json") as span:
            span.set_attribute("llm.provider", "ollama")
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.base_url", self.base_url)
            span.set_attribute("llm.temperature", float(self.temperature))
            span.set_attribute("llm.prompt_chars", len(prompt))

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": self.temperature,
                },
            }

            response = self._post_ollama(
                endpoint="/api/generate",
                payload=payload,
            )

            raw_text = response.get("response", "").strip()

            span.set_attribute("llm.raw_response_chars", len(raw_text))

            if not raw_text:
                span.set_attribute("llm.success", False)
                span.set_attribute("llm.parse_status", "empty_response")

                return {
                    "error": "empty_llm_response",
                    "raw_response": raw_text,
                }

            parsed = self._safe_parse_json(raw_text)

            span.set_attribute("llm.success", not bool(parsed.get("error")))
            span.set_attribute(
                "llm.parse_status",
                parsed.get("error") or "parsed",
            )

            return parsed

    def health_check(self) -> Dict[str, Any]:
        prompt = (
            "Return only valid JSON with this exact structure: "
            '{"status": "ok", "message": "ollama is working"}'
        )

        result = self.generate_json(prompt)

        return {
            "ollama_base_url": self.base_url,
            "model": self.model,
            "result": result,
        }

    def _post_ollama(
        self,
        endpoint: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        tracer = get_tracer()

        with tracer.start_as_current_span("eventlens.service.llm.ollama_http") as span:
            url = f"{self.base_url}{endpoint}"
            data = json.dumps(payload).encode("utf-8")

            span.set_attribute("http.method", "POST")
            span.set_attribute("http.url", url)
            span.set_attribute("llm.model", payload.get("model", self.model))
            span.set_attribute("llm.stream", bool(payload.get("stream", False)))
            span.set_attribute("llm.format", str(payload.get("format", "text")))
            span.set_attribute("llm.request_bytes", len(data))
            span.set_attribute("llm.timeout_seconds", int(self.timeout_seconds))

            request = urllib.request.Request(
                url=url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    response_body = response.read().decode("utf-8")

                    span.set_attribute("http.status_code", response.status)
                    span.set_attribute("llm.response_bytes", len(response_body))
                    span.set_attribute("llm.http_success", True)

                    return json.loads(response_body)

            except urllib.error.URLError as exc:
                span.set_attribute("llm.http_success", False)
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(exc))

                raise RuntimeError(
                    f"Failed to connect to Ollama at {url}. "
                    f"Make sure Ollama is running. Original error: {exc}"
                ) from exc

            except json.JSONDecodeError as exc:
                span.set_attribute("llm.http_success", False)
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(exc))

                raise RuntimeError(
                    f"Ollama returned non-JSON HTTP response from {url}."
                ) from exc

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        tracer = get_tracer()

        with tracer.start_as_current_span("eventlens.service.llm.parse_json") as span:
            span.set_attribute("llm.parse.input_chars", len(text))

            try:
                parsed = json.loads(text)

                span.set_attribute("llm.parse.success", True)
                span.set_attribute("llm.parse.mode", "direct_json")

                if isinstance(parsed, dict):
                    return parsed

                return {
                    "value": parsed,
                    "raw_response": text,
                }

            except json.JSONDecodeError:
                pass

            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]

                try:
                    parsed = json.loads(candidate)

                    span.set_attribute("llm.parse.success", True)
                    span.set_attribute("llm.parse.mode", "extracted_json")

                    if isinstance(parsed, dict):
                        return parsed

                    return {
                        "value": parsed,
                        "raw_response": text,
                    }

                except json.JSONDecodeError:
                    pass

            span.set_attribute("llm.parse.success", False)
            span.set_attribute("llm.parse.mode", "failed")

            return {
                "error": "json_parse_failed",
                "raw_response": text,
            }