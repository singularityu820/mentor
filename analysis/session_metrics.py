#!/usr/bin/env python3
"""Compute descriptive metrics from CaseSentinel session artefacts.

Usage
-----
python analysis/session_metrics.py outputs/sessions/（2021）晋05刑初28号_summary.json

The script summarises agent output lengths, plan coverage, and blackboard
section density across iterations. Results are written to JSON and Markdown
reports located next to the input summary (``*_metrics.json``/``*_metrics.md``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import re

WORD_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def count_words(text: str) -> int:
    """Approximate word count (handles Latin + CJK tokens)."""

    if not text:
        return 0
    return len(WORD_PATTERN.findall(text))


def parse_sections(snapshot_path: Path) -> Dict[str, str]:
    """Parse top-level Markdown sections (``##`` headers) from a snapshot."""

    text = snapshot_path.read_text(encoding="utf-8")
    sections: Dict[str, List[str]] = {}
    current: str | None = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer, current
        if current is not None:
            sections[current] = "\n".join(buffer).strip()
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            flush()
            current = line[3:].strip()
        else:
            buffer.append(line)
    flush()
    return sections


def describe(values: Iterable[float]) -> Dict[str, float]:
    """Return mean/median/stdev for non-empty lists."""

    data = list(values)
    if not data:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0}
    if len(data) == 1:
        return {"mean": data[0], "median": data[0], "stdev": 0.0}
    return {
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "stdev": statistics.stdev(data),
    }


def compute_metrics(summary_path: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    case_id: str = summary["case_id"]
    iterations = summary.get("iterations", [])

    per_iteration: List[Dict[str, object]] = []
    agent_content_words: Dict[str, List[int]] = defaultdict(list)
    agent_content_chars: Dict[str, List[int]] = defaultdict(list)
    agent_plan_words: Dict[str, List[int]] = defaultdict(list)
    agent_plan_nonempty: Dict[str, int] = defaultdict(int)
    agent_success_prob: Dict[str, List[float]] = defaultdict(list)

    section_word_counts: Dict[str, List[int]] = defaultdict(list)
    section_nonempty_counts: Dict[str, int] = defaultdict(int)

    for iteration_entry in iterations:
        iteration_index = iteration_entry["iteration"]
        outputs = iteration_entry.get("outputs", {})
        iteration_metrics: Dict[str, object] = {"iteration": iteration_index, "agents": {}}

        for agent_name, payload in outputs.items():
            content = payload.get("content") or ""
            plan = payload.get("plan") or ""
            metrics = payload.get("metrics") or {}

            content_words = count_words(content)
            content_chars = len(content)
            plan_words = count_words(plan)

            agent_content_words[agent_name].append(content_words)
            agent_content_chars[agent_name].append(content_chars)
            agent_plan_words[agent_name].append(plan_words)
            if plan.strip():
                agent_plan_nonempty[agent_name] += 1

            success_prob = metrics.get("success_probability")
            if success_prob is not None:
                try:
                    agent_success_prob[agent_name].append(float(success_prob))
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    pass

            iteration_metrics["agents"][agent_name] = {
                "content_words": content_words,
                "content_chars": content_chars,
                "plan_words": plan_words,
                "has_plan": bool(plan.strip()),
                "success_probability": success_prob,
                "knowledge_mentions": bool(re.search(r"知识检索|图谱关联", content)),
            }

        snapshot_path = summary_path.parent / f"{case_id}_iter{iteration_index}.md"
        if snapshot_path.exists():
            sections = parse_sections(snapshot_path)
            section_snapshot_metrics = {}
            for section_name, section_body in sections.items():
                words = count_words(section_body)
                section_word_counts[section_name].append(words)
                if words > 0:
                    section_nonempty_counts[section_name] += 1
                section_snapshot_metrics[section_name] = {
                    "word_count": words,
                    "char_count": len(section_body),
                }
            iteration_metrics["sections"] = section_snapshot_metrics
            iteration_metrics["snapshot_path"] = str(snapshot_path.relative_to(summary_path.parent))
        else:
            iteration_metrics["sections"] = {}
            iteration_metrics["snapshot_path"] = None

        per_iteration.append(iteration_metrics)

    iterations_count = len(iterations)
    agents = sorted(agent_content_words.keys())

    aggregated_agents = {}
    for agent_name in agents:
        coverage = agent_plan_nonempty.get(agent_name, 0) / iterations_count if iterations_count else 0.0
        aggregated_agents[agent_name] = {
            "content_words": describe(agent_content_words[agent_name]),
            "content_chars": describe(agent_content_chars[agent_name]),
            "plan_words": describe([w for w in agent_plan_words[agent_name] if w > 0]),
            "plan_coverage": coverage,
            "success_probability": describe(agent_success_prob.get(agent_name, [])),
        }

    aggregated_sections = {}
    for section_name, values in section_word_counts.items():
        aggregated_sections[section_name] = {
            "word_stats": describe(values),
            "nonempty_iterations": section_nonempty_counts.get(section_name, 0),
            "coverage": section_nonempty_counts.get(section_name, 0) / iterations_count if iterations_count else 0.0,
        }

    overview = {
        "case_id": case_id,
        "iterations": iterations_count,
        "agents": agents,
    }

    aggregated = {
        "overview": overview,
        "agents": aggregated_agents,
        "sections": aggregated_sections,
    }

    diagnostics = {
        "per_iteration": per_iteration,
    }
    return aggregated, diagnostics


def render_markdown(aggregated: Dict[str, object], diagnostics: Dict[str, object]) -> str:
    overview = aggregated["overview"]
    agents = aggregated["agents"]
    sections = aggregated["sections"]
    iterations = diagnostics["per_iteration"]

    lines = []
    lines.append(f"# Session Metrics — {overview['case_id']}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Iterations analysed: {overview['iterations']}")
    lines.append(f"- Agents: {', '.join(overview['agents']) if overview['agents'] else 'n/a'}")
    lines.append("")

    if agents:
        lines.append("## Agent Output Density")
        lines.append("")
        lines.append("| Agent | Content (words) | Content (chars) | Plan coverage | Plan (words) |")
        lines.append("| ----- | ---------------- | ---------------- | ------------- | ------------ |")
        for name, stats in agents.items():
            content_words = stats["content_words"]["mean"]
            content_chars = stats["content_chars"]["mean"]
            plan_cov = stats["plan_coverage"]
            plan_words = stats["plan_words"]["mean"] if stats["plan_words"] else 0.0
            lines.append(
                f"| {name} | {content_words:.1f} | {content_chars:.1f} | {plan_cov:.2%} | {plan_words:.1f} |"
            )
        lines.append("")

    if sections:
        lines.append("## Blackboard Section Coverage")
        lines.append("")
        lines.append("| Section | Coverage | Mean words | Median words |")
        lines.append("| ------- | -------- | ---------- | ------------ |")
        for section_name, stats in sorted(sections.items()):
            word_stats = stats["word_stats"]
            coverage = stats["coverage"]
            lines.append(
                f"| {section_name} | {coverage:.2%} | {word_stats['mean']:.1f} | {word_stats['median']:.1f} |"
            )
        lines.append("")

    if iterations:
        lines.append("## Iteration Details")
        lines.append("")
        lines.append("| Iteration | Analyst words | Strategist words | Forecaster words | Snapshot |")
        lines.append("| --------- | ------------- | ---------------- | ---------------- | -------- |")
        for entry in iterations:
            idx = entry["iteration"]
            agents_info = entry.get("agents", {})
            analyst_words = agents_info.get("analyst", {}).get("content_words", 0)
            strategist_words = agents_info.get("strategist", {}).get("content_words", 0)
            forecaster_words = agents_info.get("forecaster", {}).get("content_words", 0)
            snapshot = entry.get("snapshot_path") or "-"
            lines.append(
                f"| {idx} | {analyst_words} | {strategist_words} | {forecaster_words} | {snapshot} |"
            )
        lines.append("")

    return "\n".join(lines)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Path to the *_summary.json file")
    parser.add_argument("--json", dest="json_output", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--markdown", dest="markdown_output", type=Path, help="Optional Markdown output path"
    )
    args = parser.parse_args(argv[1:])

    aggregated, diagnostics = compute_metrics(args.summary)

    json_output = args.json_output or args.summary.with_name(args.summary.stem + "_metrics.json")
    markdown_output = args.markdown_output or args.summary.with_name(args.summary.stem + "_metrics.md")

    json_output.write_text(json.dumps({"aggregated": aggregated, "diagnostics": diagnostics}, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_output.write_text(render_markdown(aggregated, diagnostics), encoding="utf-8")

    print(f"Metrics written to {json_output} and {markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
