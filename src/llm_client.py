"""One small swappable interface to whichever LLM we call.

Two implementations: a Gemini API client (the default system-side path) and a
stub for a local vLLM endpoint on Leonardo (the open-source path).

WALL RULE (model-family split): the personas are played by Gemma. Anything on
the system side of the Wall -- interviewer, item encoder, person encoder -- must
run on a different model family, so the same weights are never on both sides.
Do not point a system-side client at a Gemma endpoint.

Credentials come from the environment, read at call time, never hardcoded:
  GLASSBOX_API_KEY  the API key
  MODEL_NAME        e.g. the flash-lite tier model id
The key is never logged and never included in an exception message.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass
class LLMResponse:
    """One completion plus its token counts."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient(ABC):
    """Anything that can turn a prompt into text."""

    @abstractmethod
    def generate(
        self,
        system: str | None,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Return one completion for ``user``, optionally steered by ``system``."""
        raise NotImplementedError


def load_dotenv_if_present(path: str | Path | None = None) -> list[str]:
    """Load ``KEY=value`` lines from a ``.env`` file into ``os.environ``.

    Existing environment variables win, so a real shell export is never
    clobbered. Returns the names of the variables that were set. Missing file is
    not an error. Values may be wrapped in single or double quotes; lines that
    are blank, comments, or have no ``=`` are skipped.
    """
    env_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return []

    loaded: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


class GeminiClient(LLMClient):
    """Google Generative Language REST API (v1beta ``:generateContent``)."""

    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout

    @staticmethod
    def _require(name: str) -> str:
        value = os.environ.get(name, "").strip()
        if not value:
            raise RuntimeError(
                f"environment variable {name} is not set -- put it in the "
                f"gitignored .env at the repo root and call "
                f"load_dotenv_if_present() first"
            )
        return value

    def generate(
        self,
        system: str | None,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        api_key = self._require("GLASSBOX_API_KEY")
        model = self._require("MODEL_NAME")

        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        request = urllib.request.Request(
            # The key goes in a header, never in the URL: urllib puts the URL
            # into HTTPError messages and tracebacks.
            url=_GEMINI_ENDPOINT.format(model=model),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Report status and the API's own message; never echo the request.
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"Gemini API returned HTTP {exc.code} for model {model}: {detail}"
            ) from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach the Gemini API: {exc.reason}") from None

        return LLMResponse(
            text=_first_text(body),
            input_tokens=_usage(body, "promptTokenCount"),
            output_tokens=_usage(body, "candidatesTokenCount"),
            model=model,
        )


def _first_text(body: dict) -> str:
    """Concatenate the text parts of the first candidate; empty string if none."""
    candidates = body.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts)


def _usage(body: dict, field: str) -> int:
    return int((body.get("usageMetadata") or {}).get(field, 0))


class LocalVLLMClient(LLMClient):
    """Placeholder for the open-source path.

    Will target an OpenAI-compatible vLLM server running a dense model on
    Leonardo (``/v1/chat/completions``). Not built yet: whether we use the API
    path or the local path is the Gate 0 decision, still pending.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or os.environ.get("GLASSBOX_VLLM_URL", "")

    def generate(
        self,
        system: str | None,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        raise NotImplementedError(
            "the local vLLM path is not built yet -- pending the Gate 0 decision "
            "on API vs self-hosted"
        )


_CLIENTS: dict[str, type[LLMClient]] = {
    "gemini": GeminiClient,
    "vllm": LocalVLLMClient,
}


def get_client(name: str = "gemini") -> LLMClient:
    """Return a client by name: ``gemini`` or ``vllm``."""
    try:
        return _CLIENTS[name]()
    except KeyError:
        raise ValueError(
            f"unknown client {name!r}; known: {', '.join(sorted(_CLIENTS))}"
        ) from None
