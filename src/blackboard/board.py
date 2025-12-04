"""In-memory representation of the CrimeMentor central blackboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

DEFAULT_TEMPLATE_PATH = Path(__file__).with_name("blackboard_template.md")


@dataclass
class Blackboard:
    """Markdown-based collaborative workspace."""

    template_path: Path = DEFAULT_TEMPLATE_PATH
    sections: Dict[str, str] = field(default_factory=dict)

    def load_template(self) -> None:
        content = self.template_path.read_text(encoding="utf-8")
        current_section = "root"
        self.sections = {current_section: []}
        for line in content.splitlines():
            if line.startswith("# "):
                current_section = line.strip("# ").strip()
                self.sections[current_section] = []
            else:
                self.sections[current_section].append(line)
        # Join lines back to strings
        self.sections = {key: "\n".join(lines).strip() for key, lines in self.sections.items()}

    def snapshot(self) -> str:
        if not self.sections:
            self.load_template()
        chunks = []
        for title, body in self.sections.items():
            if title == "root":
                continue
            chunks.append(f"## {title}\n{body}\n")
        return "\n".join(chunks)

    def update_section(self, title: str, content: str) -> None:
        if not self.sections:
            self.load_template()
        self.sections[title] = content.strip()

    def append_to_section(self, title: str, content: str) -> None:
        if not self.sections:
            self.load_template()
        existing = self.sections.get(title, "")
        combined = (existing + "\n" + content).strip() if existing else content.strip()
        self.sections[title] = combined
