"""Generate self-contained HTML reports from session outputs."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ReportConfig:
    session_dir: Path
    case_id: Optional[str] = None
    output_path: Optional[Path] = None


class SessionReportGenerator:
    def __init__(self, config: ReportConfig) -> None:
        self.config = config

    def generate(self) -> Path:
        summary_path, case_id = self._resolve_summary()
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        case_id = data["case_id"]

        output_path = self.config.output_path or (summary_path.with_suffix(".html"))
        html_content = self._build_html(case_id, data)
        output_path.write_text(html_content, encoding="utf-8")
        return output_path

    def _resolve_summary(self) -> tuple[Path, str]:
        session_dir = self.config.session_dir
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory {session_dir} does not exist")

        if self.config.case_id:
            summary_path = session_dir / f"{self.config.case_id}_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Summary file {summary_path} not found")
            return summary_path, self.config.case_id

        summaries = list(session_dir.glob("*_summary.json"))
        if not summaries:
            raise FileNotFoundError("No summary json found in session directory")
        summaries.sort()
        chosen = summaries[-1]
        case_id = chosen.stem.replace("_summary", "")
        return chosen, case_id

    def _build_html(self, case_id: str, data: Dict[str, object]) -> str:
        iterations = data.get("iterations", [])
        sections = []
        for entry in iterations:
            iteration_idx = entry.get("iteration")
            outputs = entry.get("outputs", {})
            agent_sections = []
            for agent_name, payload in outputs.items():
                content = html.escape(payload.get("content", ""))
                plan = html.escape(payload.get("plan") or "(无)")
                metrics_html = "<ul>" + "".join(
                    f"<li>{html.escape(k)}: {html.escape(str(v))}</li>" for k, v in (payload.get("metrics", {}) or {}).items()
                ) + "</ul>"
                agent_sections.append(
                    f"<div class='agent'><h4>{agent_name.title()}</h4>"
                    f"<h5>Plan</h5><pre>{plan}</pre>"
                    f"<h5>Content</h5><pre>{content}</pre>"
                    f"<h5>Metrics</h5>{metrics_html}</div>"
                )
            snapshot_path = self.config.session_dir / f"{case_id}_iter{iteration_idx}.md"
            snapshot_text = html.escape(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.exists() else ""
            sections.append(
                f"<section><h2>Iteration {iteration_idx}</h2>"
                f"<details open><summary>Blackboard Snapshot</summary><pre>{snapshot_text}</pre></details>"
                + "".join(agent_sections)
                + "</section>"
            )

        sections_html = "".join(sections)
        return f"""
<!DOCTYPE html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <title>CrimeMentor Session Report - {html.escape(case_id)}</title>
  <style>
    body {{ font-family: 'SF Pro Display', 'Microsoft Yahei', sans-serif; margin: 2rem; background: #f5f7fa; }}
    header {{ margin-bottom: 2rem; }}
    section {{ background: white; padding: 1.5rem; margin-bottom: 1.5rem; border-radius: 12px; box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08); }}
    h1 {{ color: #0f172a; }}
    pre {{ background: #0f172a; color: #e2e8f0; padding: 1rem; border-radius: 8px; overflow-x: auto; }}
    .agent {{ border-top: 1px solid #e2e8f0; padding-top: 1rem; margin-top: 1rem; }}
    details {{ margin-bottom: 1rem; }}
    summary {{ cursor: pointer; font-weight: bold; }}
  </style>
</head>
<body>
  <header>
    <h1>CrimeMentor Session Report</h1>
    <h2>Case ID: {html.escape(case_id)}</h2>
    <p>Iterations: {len(iterations)}</p>
  </header>
  {sections_html}
</body>
</html>
"""


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate an HTML report from session outputs")
    parser.add_argument("session_dir", type=Path, help="Directory containing session artifacts")
    parser.add_argument("--case-id", help="Case identifier")
    parser.add_argument("--output", type=Path, help="Output HTML path")
    args = parser.parse_args()

    generator = SessionReportGenerator(
        ReportConfig(
            session_dir=args.session_dir,
            case_id=args.case_id,
            output_path=args.output,
        )
    )
    path = generator.generate()
    print(f"Report saved to {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
