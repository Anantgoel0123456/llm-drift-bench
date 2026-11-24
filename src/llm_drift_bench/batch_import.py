"""Offline importers converting OpenAI Batch, Anthropic Batch, and CSV files to ModelSamples."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from llm_drift_bench._file_io import parse_source_json, read_utf8
from llm_drift_bench.models import ModelSample


def _iter_jsonl_objects(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    source = Path(path)
    raw = read_utf8(source)
    rows: list[tuple[int, dict[str, Any]]] = []
    for lineno, line in enumerate(raw.split("\n"), start=1):
        if not line.strip(" \t\r"):
            continue
        parsed = parse_source_json(line, source, line=lineno)
        if not isinstance(parsed, dict):
            raise ValueError(f"{source} (line {lineno}): batch record must be a JSON object.")
        rows.append((lineno, parsed))
    if not rows:
        raise ValueError(f"{source}: batch output file is empty.")
    return rows


def _extract_openai_body_text_and_tokens(
    body: dict[str, Any],
    *,
    where: str,
) -> tuple[str, int | None]:
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError(f"{where}: choice entry must be an object.")
        msg = first.get("message")
        if not isinstance(msg, dict):
            raise ValueError(f"{where}: missing choices[0].message object.")
        raw_content = msg.get("content")
        if isinstance(raw_content, str):
            text_out = raw_content
        elif isinstance(raw_content, list):
            content_parts = [
                block["text"]
                for block in raw_content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            if not content_parts:
                raise ValueError(f"{where}: choices[0].message.content list has no text blocks.")
            text_out = "".join(content_parts)
        else:
            raise ValueError(f"{where}: missing string or text-block choices[0].message.content.")
        usage = body.get("usage")
        tokens: int | None = None
        if isinstance(usage, dict):
            raw_toks = usage.get("completion_tokens")
            if type(raw_toks) is int and raw_toks >= 0:
                tokens = raw_toks
        return text_out, tokens

    # Support OpenAI Responses API batch output bodies too.
    output_items = body.get("output")
    if isinstance(output_items, list) and output_items:
        parts: list[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
        if parts:
            usage = body.get("usage")
            tokens = None
            if isinstance(usage, dict):
                raw_toks = usage.get("output_tokens")
                if type(raw_toks) is int and raw_toks >= 0:
                    tokens = raw_toks
            return "".join(parts), tokens

    raise ValueError(f"{where}: response.body contains neither chat 'choices' nor 'output' text.")


def import_openai_batch_jsonl(
    path: str | Path,
    *,
    model_name: str | None = None,
) -> list[ModelSample]:
    """Import an OpenAI Batch API output .jsonl file into canonical ModelSamples."""
    source = Path(path)
    samples: list[ModelSample] = []
    seen_ids: set[str] = set()

    for lineno, row in _iter_jsonl_objects(source):
        where = f"{source} (line {lineno})"
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id.strip():
            raise ValueError(f"{where}: missing non-empty string 'custom_id'.")
        if custom_id in seen_ids:
            raise ValueError(f"{where}: duplicate custom_id {custom_id!r}.")
        seen_ids.add(custom_id)

        if row.get("error") is not None:
            raise ValueError(f"{where}: batch item {custom_id!r} failed with an API error.")
        resp = row.get("response")
        if not isinstance(resp, dict):
            raise ValueError(f"{where}: batch item {custom_id!r} is missing 'response' object.")
        status_code = resp.get("status_code")
        if status_code != 200:
            raise ValueError(
                f"{where}: batch item {custom_id!r} returned HTTP status {status_code!r}."
            )
        body = resp.get("body")
        if not isinstance(body, dict):
            raise ValueError(f"{where}: batch item {custom_id!r} is missing 'response.body'.")

        text, tokens = _extract_openai_body_text_and_tokens(body, where=where)
        body_model = body.get("model")
        resolved_model = (
            model_name
            or (str(body_model) if isinstance(body_model, str) and body_model else None)
            or "openai-batch"
        )
        meta: dict[str, Any] = {"source_format": "openai_batch"}
        if isinstance(row.get("id"), str):
            meta["batch_request_id"] = row["id"]
        samples.append(
            ModelSample(
                prompt_id=custom_id,
                output=text,
                model_name=resolved_model,
                latency_ms=None,
                tokens_generated=tokens,
                metadata=meta,
            )
        )
    return sorted(samples, key=lambda s: s.prompt_id)


def import_anthropic_batch_jsonl(
    path: str | Path,
    *,
    model_name: str | None = None,
) -> list[ModelSample]:
    """Import an Anthropic Message Batches output .jsonl file into canonical ModelSamples."""
    source = Path(path)
    samples: list[ModelSample] = []
    seen_ids: set[str] = set()

    for lineno, row in _iter_jsonl_objects(source):
        where = f"{source} (line {lineno})"
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id.strip():
            raise ValueError(f"{where}: missing non-empty string 'custom_id'.")
        if custom_id in seen_ids:
            raise ValueError(f"{where}: duplicate custom_id {custom_id!r}.")
        seen_ids.add(custom_id)

        result = row.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"{where}: batch item {custom_id!r} is missing 'result' object.")
        res_type = result.get("type")
        if res_type != "succeeded":
            raise ValueError(
                f"{where}: batch item {custom_id!r} did not succeed (result.type={res_type!r})."
            )
        message = result.get("message")
        if not isinstance(message, dict):
            raise ValueError(f"{where}: batch item {custom_id!r} is missing 'result.message'.")
        blocks = message.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(f"{where}: batch item {custom_id!r} has empty message.content.")
        text_parts = [
            block["text"]
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if not text_parts:
            raise ValueError(f"{where}: batch item {custom_id!r} contains no text content blocks.")

        usage = message.get("usage")
        tokens: int | None = None
        if isinstance(usage, dict):
            raw_toks = usage.get("output_tokens")
            if type(raw_toks) is int and raw_toks >= 0:
                tokens = raw_toks
        resolved_model = (
            model_name
            or (
                str(message["model"])
                if isinstance(message.get("model"), str) and message["model"]
                else None
            )
            or "anthropic-batch"
        )
        samples.append(
            ModelSample(
                prompt_id=custom_id,
                output="".join(text_parts),
                model_name=resolved_model,
                latency_ms=None,
                tokens_generated=tokens,
                metadata={"source_format": "anthropic_batch"},
            )
        )
    return sorted(samples, key=lambda s: s.prompt_id)


def import_csv_captures(
    path: str | Path,
    *,
    prompt_id_col: str = "prompt_id",
    output_col: str = "output",
    model_col: str = "model_name",
    latency_col: str = "latency_ms",
    tokens_col: str = "tokens_generated",
    default_model: str = "unknown",
) -> list[ModelSample]:
    """Import captured outputs from a UTF-8 CSV file into canonical ModelSamples."""
    source = Path(path)
    raw = read_utf8(source)
    reader = csv.DictReader(io.StringIO(raw))
    fieldnames = reader.fieldnames or []
    if prompt_id_col not in fieldnames or output_col not in fieldnames:
        raise ValueError(
            f"{source}: CSV must include header columns {prompt_id_col!r} and {output_col!r}."
        )

    samples: list[ModelSample] = []
    seen_ids: set[str] = set()
    for row_idx, row in enumerate(reader, start=2):
        where = f"{source} (row {row_idx})"
        pid = (row.get(prompt_id_col) or "").strip()
        if not pid:
            raise ValueError(f"{where}: empty {prompt_id_col!r} value.")
        if pid in seen_ids:
            raise ValueError(f"{where}: duplicate prompt_id {pid!r}.")
        seen_ids.add(pid)

        output = row.get(output_col)
        if output is None:
            raise ValueError(f"{where}: missing {output_col!r} cell.")
        model = (row.get(model_col) or "").strip() or default_model

        latency: float | None = None
        raw_lat = (row.get(latency_col) or "").strip()
        if raw_lat:
            try:
                latency = float(raw_lat)
            except ValueError as exc:
                raise ValueError(f"{where}: invalid {latency_col!r} value {raw_lat!r}.") from exc

        tokens: int | None = None
        raw_toks = (row.get(tokens_col) or "").strip()
        if raw_toks:
            try:
                tokens = int(raw_toks)
            except ValueError as exc:
                raise ValueError(f"{where}: invalid {tokens_col!r} value {raw_toks!r}.") from exc

        try:
            samples.append(
                ModelSample(
                    prompt_id=pid,
                    output=output,
                    model_name=model,
                    latency_ms=latency,
                    tokens_generated=tokens,
                )
            )
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
    if not samples:
        raise ValueError(f"{source}: CSV contains zero data rows.")
    return sorted(samples, key=lambda s: s.prompt_id)
