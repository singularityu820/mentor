"""CLI helper for generating detective-style narratives from cleaned cases."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Set

from .cleaner import CaseRecord
from .narrative_generator import NarrativeGenerator

LOGGER = logging.getLogger(__name__)


def _load_case_records(path: Path, limit: Optional[int] = None) -> Iterator[CaseRecord]:
    with path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            yield CaseRecord.model_validate(payload)
            if limit is not None and idx >= limit:
                break


def _load_processed_case_ids(path: Path) -> Set[str]:
    processed: Set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    case_id = data.get("case_id")
                    if isinstance(case_id, str):
                        processed.add(case_id)
                except json.JSONDecodeError:
                    LOGGER.warning("Skipping malformed JSONL line in %s", path)
    except FileNotFoundError:
        pass
    return processed


def _load_failed_case_ids(path: Path) -> Set[str]:
    failed: Set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("status", "failed") != "failed":
                        continue
                    case_id = data.get("case_id")
                    if isinstance(case_id, str):
                        failed.add(case_id)
                except json.JSONDecodeError:
                    LOGGER.warning("Skipping malformed failure log line in %s", path)
    except FileNotFoundError:
        pass
    return failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate high-density narratives for cleaned case records.",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("outputs/cleaned_cases.jsonl"),
        help="Path to the cleaned cases JSONL file (default: outputs/cleaned_cases.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/narratives.jsonl"),
        help="Destination JSONL path for generated narratives (default: outputs/narratives.jsonl)",
    )
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        help="Optional directory to dump each narrative as an individual Markdown file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most N cases (useful for dry runs).",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume generation by skipping cases already present in the output JSONL.",
    )
    parser.add_argument(
        "--failed-log",
        type=Path,
        default=Path("outputs/narratives_failed.jsonl"),
        help="Record failed generations to this JSONL file (default: outputs/narratives_failed.jsonl)",
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="When resuming, also skip cases that appear in the failed log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log.upper())

    LOGGER.info("Loading cases from %s", args.input)
    record_iter = _load_case_records(args.input, args.limit)

    processed_ids: Set[str] = set()
    failed_ids: Set[str] = set()
    file_mode = "w"
    if args.resume:
        if args.output.exists():
            processed_ids = _load_processed_case_ids(args.output)
            file_mode = "a"
            LOGGER.info("Resuming run with %d existing narratives", len(processed_ids))
        else:
            LOGGER.info("Resume flag set but %s not found; starting fresh.", args.output)

    if args.skip_failures and args.failed_log:
        failed_ids = _load_failed_case_ids(args.failed_log)
        if failed_ids:
            LOGGER.info("Loaded %d failed case ids to skip", len(failed_ids))

    generator = NarrativeGenerator()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    markdown_dir = args.markdown_dir
    if markdown_dir:
        markdown_dir.mkdir(parents=True, exist_ok=True)

    skip_ids = processed_ids | failed_ids
    if skip_ids:
        LOGGER.info(
            "Skipping %d cases already handled (%d successes, %d failures)",
            len(skip_ids),
            len(processed_ids),
            len(failed_ids),
        )

    written = 0
    md_written = 0

    LOGGER.info("Generating narratives using live LLM client")
    with args.output.open(file_mode, encoding="utf-8") as fp:
        for record in record_iter:
            if skip_ids and record.case_id in skip_ids:
                LOGGER.debug("Skipping case %s (already processed)", record.case_id)
                continue

            try:
                sample = generator.generate(record)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Narrative generation failed for %s: %s", record.case_id, exc)
                if args.failed_log:
                    args.failed_log.parent.mkdir(parents=True, exist_ok=True)
                    failure_payload = {
                        "case_id": record.case_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "error": str(exc),
                        "status": "failed",
                        "provider": getattr(generator.llm_client, "provider", "unknown"),
                    }
                    with args.failed_log.open("a", encoding="utf-8") as failure_fp:
                        failure_fp.write(json.dumps(failure_payload, ensure_ascii=False) + "\n")
                    failed_ids.add(record.case_id)
                    skip_ids.add(record.case_id)
                continue

            fp.write(json.dumps(sample.model_dump(mode="python"), ensure_ascii=False) + "\n")
            fp.flush()
            written += 1

            if args.failed_log and record.case_id in failed_ids:
                recovery_payload = {
                    "case_id": record.case_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "status": "recovered",
                }
                with args.failed_log.open("a", encoding="utf-8") as failure_fp:
                    failure_fp.write(json.dumps(recovery_payload, ensure_ascii=False) + "\n")
                failed_ids.remove(record.case_id)
                skip_ids.discard(record.case_id)

            if markdown_dir:
                markdown_path = markdown_dir / f"{sample.case_id}.md"
                if args.resume and markdown_path.exists():
                    LOGGER.debug("Skipping existing markdown for %s", sample.case_id)
                else:
                    markdown_path.write_text(
                        NarrativeGenerator.to_markdown(sample),
                        encoding="utf-8",
                    )
                    md_written += 1

    if processed_ids and written == 0:
        LOGGER.info("No new cases were processed; existing output already complete.")
        return

    if written == 0:
        LOGGER.warning("No case records loaded; aborting narrative generation.")
        return

    LOGGER.info("Narrative generation complete: %d new JSONL records, %d Markdown files", written, md_written)


if __name__ == "__main__":  # pragma: no cover
    main()
