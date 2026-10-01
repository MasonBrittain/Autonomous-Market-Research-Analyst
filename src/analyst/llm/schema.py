"""Turn a Pydantic model into a JSON Schema the API will accept as an output format.

Pydantic emits `$ref`/`$defs` and omits `additionalProperties`, neither of which
suits a strict output format, so references are inlined and every object is closed
and fully required. Optional fields stay expressible via `anyOf [T, null]`.

Keeping this explicit (rather than leaning on `messages.parse`) is what lets a
single request carry both `output_config.format` and `output_config.effort`, which
is how per-node effort tiering is implemented.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# Keys that carry no constraint and only cost tokens in the request.
_STRIP = {"title", "default", "examples", "$comment", "readOnly", "writeOnly"}


def _resolve(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    if depth > 24:
        raise ValueError("schema nesting too deep to inline; flatten the output model")

    if isinstance(node, list):
        return [_resolve(item, defs, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref = node["$ref"]
        name = ref.rsplit("/", 1)[-1]
        if name not in defs:
            raise ValueError(f"unresolvable schema reference: {ref}")
        merged = _resolve(defs[name], defs, depth + 1)
        # Sibling keys alongside $ref (e.g. a description) win over the target.
        extra = {k: v for k, v in node.items() if k != "$ref" and k not in _STRIP}
        if isinstance(merged, dict):
            return {**merged, **extra}
        return merged

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _STRIP or key == "$defs":
            continue
        out[key] = _resolve(value, defs, depth + 1)

    if out.get("type") == "object" or "properties" in out:
        out.setdefault("type", "object")
        properties = out.get("properties") or {}
        out["properties"] = properties
        # Strict output formats require every property to be required and the
        # object closed. Optionality is expressed by allowing null, not by absence.
        out["required"] = list(properties.keys())
        out["additionalProperties"] = False

    return out


def to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})
    schema = _resolve(raw, defs)
    if not isinstance(schema, dict):
        raise TypeError("model schema did not resolve to an object")
    schema.setdefault("type", "object")
    return schema


def output_format_for(model: type[BaseModel]) -> dict[str, Any]:
    return {"type": "json_schema", "schema": to_strict_schema(model)}
