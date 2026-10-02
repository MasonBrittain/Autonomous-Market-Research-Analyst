"""The only module that talks to the Claude API.

Design notes worth keeping:

* Every call returns a `UsageRecord` alongside the parsed result, so cost is
  accounted per node from the first request rather than estimated afterwards.
* The system prompt is a list of blocks with explicit cache flags. Caching is a
  prefix match, so the stable part (instructions + fact pack) is cached and only
  the per-call instruction varies after the breakpoint. `cache_read_tokens` on the
  ledger is the proof that this is working -- if it stays at zero, something
  volatile has leaked into the prefix.
* `StubLLM` implements the same protocol with no network, so the whole pipeline is
  runnable and testable without an API key.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel, ValidationError

from ..models import UsageRecord
from .schema import output_format_for

if TYPE_CHECKING:  # the SDK's request types, for checking only
    from anthropic.types import (
        Message,
        MessageParam,
        OutputConfigParam,
        ParsedMessage,
        TextBlockParam,
    )

Effort = Literal["low", "medium", "high", "xhigh", "max"]
VALID_EFFORT: frozenset[str] = frozenset(("low", "medium", "high", "xhigh", "max"))


def coerce_effort(value: str) -> Effort:
    """Map a configured effort string onto the documented set.

    `RunConfig.effort` comes from JSON, so a typo is possible; defaulting to the
    API's own default is better than sending a value it will reject.
    """
    cleaned = value.strip().lower()
    return cast("Effort", cleaned) if cleaned in VALID_EFFORT else "high"


T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    """The model declined the request. Surfaced rather than retried blindly."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(f"model refused (category={category}): {explanation}")
        self.category = category
        self.explanation = explanation


@dataclass
class SystemBlock:
    """One block of the system prompt.

    `cache=True` marks a cache breakpoint after this block. Order matters: blocks
    are rendered in sequence and the cached prefix is everything up to the last
    flagged block, so stable content must come first.
    """

    text: str
    cache: bool = False


@dataclass
class LLMResult(Generic[T]):
    parsed: T
    usage: UsageRecord
    raw_text: str = ""


@dataclass
class CallStats:
    calls: int = 0
    refusals: int = 0
    validation_retries: int = 0
    by_node: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    """Implemented by `AnthropicLLM` and `StubLLM`."""

    model: str

    def structured(
        self,
        *,
        node: str,
        system: list[SystemBlock] | str,
        user: str,
        output_model: type[T],
        effort: str = "high",
        max_tokens: int = 8000,
    ) -> LLMResult[T]: ...


def _system_param(system: list[SystemBlock] | str) -> list[TextBlockParam]:
    blocks = [SystemBlock(system, cache=True)] if isinstance(system, str) else system
    out: list[TextBlockParam] = []
    for block in blocks:
        if not block.text.strip():
            continue
        entry: TextBlockParam = {"type": "text", "text": block.text}
        if block.cache:
            entry["cache_control"] = {"type": "ephemeral"}
        out.append(entry)
    return out


class AnthropicLLM:
    """Real client. Structured output and effort are sent in one `output_config`."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        *,
        max_retries: int = 3,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.max_retries = max_retries
        self.stats = CallStats()
        # Set once the API rejects a hand-built schema, so we stop trying it.
        self._use_sdk_parse = False

    # -- public surface ---------------------------------------------------- #

    def structured(
        self,
        *,
        node: str,
        system: list[SystemBlock] | str,
        user: str,
        output_model: type[T],
        effort: str = "high",
        max_tokens: int = 8000,
    ) -> LLMResult[T]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._attempt(
                    node=node,
                    system=system,
                    user=user,
                    output_model=output_model,
                    effort=effort,
                    max_tokens=max_tokens,
                    repair=attempt > 0,
                )
            except LLMRefusal:
                self.stats.refusals += 1
                raise
            except ValidationError as exc:
                # The response was valid JSON but not a valid instance. Retry once
                # with the validation error appended so the model can self-correct.
                self.stats.validation_retries += 1
                last_error = exc
                user = f"{user}\n\nYour previous response failed validation:\n{exc}\nReturn corrected JSON."
            except self._anthropic.RateLimitError as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
            except self._anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_error = exc
                    time.sleep(min(2**attempt, 8))
                else:
                    raise
            except self._anthropic.APIConnectionError as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise LLMError(f"{node}: exhausted retries") from last_error

    def count_tokens(self, system: str, user: str) -> int:
        """Measured, not estimated -- never approximate this with a tokenizer."""
        result = self.client.messages.count_tokens(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return int(result.input_tokens)

    # -- internals --------------------------------------------------------- #

    def _attempt(
        self,
        *,
        node: str,
        system: list[SystemBlock] | str,
        user: str,
        output_model: type[T],
        effort: str,
        max_tokens: int,
        repair: bool,
    ) -> LLMResult[T]:
        started = time.monotonic()
        system_param = _system_param(system)
        messages: list[MessageParam] = [{"role": "user", "content": user}]

        # `parse` returns a ParsedMessage (carrying `parsed_output`); `create` with an
        # output_config format returns a plain Message whose first text block is valid
        # JSON. Both paths are handled below.
        response: ParsedMessage[T] | Message

        if self._use_sdk_parse:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system_param,
                messages=messages,
                output_format=output_model,
                thinking={"type": "adaptive"},
            )
        else:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_param,
                    messages=messages,
                    thinking={"type": "adaptive"},
                    output_config=cast(
                        "OutputConfigParam",
                        {
                            "effort": coerce_effort(effort),
                            "format": output_format_for(output_model),
                        },
                    ),
                )
            except self._anthropic.BadRequestError:
                # Fall back to SDK-managed schema generation, permanently.
                self._use_sdk_parse = True
                response = self.client.messages.parse(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_param,
                    messages=messages,
                    output_format=output_model,
                    thinking={"type": "adaptive"},
                )

        latency_ms = int((time.monotonic() - started) * 1000)
        usage = self._usage(node, response, latency_ms)

        # Always check stop_reason before reading content.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(
                getattr(details, "category", None), getattr(details, "explanation", None)
            )

        parsed = getattr(response, "parsed_output", None)
        if isinstance(parsed, output_model):
            self._record(node)
            return LLMResult(parsed=parsed, usage=usage, raw_text="")

        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break
        if not text.strip():
            raise LLMError(f"{node}: empty response (stop_reason={response.stop_reason})")
        instance = output_model.model_validate(json.loads(_strip_fences(text)))
        self._record(node)
        return LLMResult(parsed=instance, usage=usage, raw_text=text)

    def _record(self, node: str) -> None:
        self.stats.calls += 1
        self.stats.by_node[node] = self.stats.by_node.get(node, 0) + 1

    def _usage(self, node: str, response: Any, latency_ms: int) -> UsageRecord:
        usage = getattr(response, "usage", None)
        return UsageRecord(
            node=node,
            model=getattr(response, "model", self.model) or self.model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            latency_ms=latency_ms,
            stop_reason=getattr(response, "stop_reason", None),
        )


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def build_client(model: str, api_key: str | None, *, stub: bool = False) -> LLMClient:
    from .stub import StubLLM

    if stub or not api_key:
        return StubLLM(model=model)
    return AnthropicLLM(model=model, api_key=api_key)
