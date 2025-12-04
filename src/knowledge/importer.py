"""Knowledge base importer for CrimeMentor.

This script ingests structured案件知识 into a vector database + 图数据库, so
that agents can perform高效的检索增强推理 (RAG) 与关系推理。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import networkx as nx
import chromadb
from chromadb.config import Settings

from src.data.cleaner import CaseRecord, load_raw_cases, clean_case_document
from src.data import LLMCaseCleaner


def load_cleaned_records(path: Path) -> List[CaseRecord]:
    records: List[CaseRecord] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(CaseRecord.model_validate(data))
    return records

LOGGER = logging.getLogger(__name__)


def build_vector_client(persist_dir: Path) -> chromadb.Client:
    persist_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(chromadb, "PersistentClient"):
        try:
            return chromadb.PersistentClient(path=str(persist_dir))
        except TypeError:  # pragma: no cover - defensive fallback for API mismatches
            LOGGER.warning("PersistentClient initialisation encountered TypeError; falling back to legacy client")

    try:
        return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_dir)))
    except ValueError as exc:
        raise RuntimeError(
            "Failed to initialise Chroma client. Please upgrade to the latest Chroma or remove existing"
            " legacy configuration. More info: https://docs.trychroma.com/deployment/migration"
        ) from exc


def upsert_vector_records(collection, records: Iterable[CaseRecord]) -> None:
    for record in records:
        doc = record.factual_findings or record.proceedings_summary
        metadata = {
            "case_id": record.case_id,
            "charges": "、".join(record.charges),
            "court": record.court or "",
        }
        collection.upsert(
            documents=[doc],
            ids=[record.case_id],
            metadatas=[metadata],
        )


def build_case_graph(records: Iterable[CaseRecord]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for record in records:
        graph.add_node(record.case_id, type="case", court=record.court, charges=record.charges)
        for participant in record.participants:
            person_node = f"{participant.role}:{participant.name}"
            graph.add_node(person_node, type="participant", role=participant.role)
            graph.add_edge(person_node, record.case_id, relation="involved_in")
        for evidence in record.evidence:
            evidence_node = f"evidence:{record.case_id}:{evidence.evidence_id}"
            graph.add_node(evidence_node, type="evidence", summary=evidence.summary)
            graph.add_edge(evidence_node, record.case_id, relation="belongs_to")
    return graph


def save_graph(graph: nx.MultiDiGraph, output_path: Path) -> None:
    data = nx.node_link_data(graph, edges="edges")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def ingest(
    input_path: Path,
    persist_dir: Path,
    graph_output: Path,
    *,
    mode: str = "rule",
    cleaned_jsonl: Optional[Path] = None,
) -> None:
    if cleaned_jsonl:
        if not cleaned_jsonl.exists():
            raise FileNotFoundError(f"Cleaned cases file not found: {cleaned_jsonl}")
        LOGGER.info("Loading pre-cleaned cases from %s", cleaned_jsonl)
        records = load_cleaned_records(cleaned_jsonl)
    else:
        raw_entries = list(load_raw_cases(input_path))

        if mode == "rule":
            records = [clean_case_document(entry) for entry in raw_entries]
        elif mode == "llm":
            llm_cleaner = LLMCaseCleaner()
            records = [llm_cleaner.clean(entry) for entry in raw_entries]
        elif mode == "hybrid":
            llm_cleaner = LLMCaseCleaner()
            records = []
            for entry in raw_entries:
                fallback = clean_case_document(entry)
                records.append(llm_cleaner.clean(entry, fallback_record=fallback))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    client = build_vector_client(persist_dir)
    collection = client.get_or_create_collection(name="crime_cases")
    upsert_vector_records(collection, records)
    persist_callable = getattr(client, "persist", None)
    if callable(persist_callable):
        persist_callable()
    else:
        LOGGER.debug("Chroma client does not expose persist(); relying on automatic persistence.")

    graph = build_case_graph(records)
    save_graph(graph, graph_output)
    LOGGER.info("Imported %d cases into knowledge base", len(records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import cleaned case data into vector and graph stores.")
    parser.add_argument("input", type=Path, help="Path to the raw data.json file")
    parser.add_argument("--cleaned-jsonl", type=Path, help="Optional path to pre-cleaned CaseRecord JSONL (skip re-cleaning)")
    parser.add_argument("--persist", type=Path, default=Path("./knowledge_store"), help="Vector store directory")
    parser.add_argument("--graph", type=Path, default=Path("./outputs/case_graph.json"), help="Graph output path")
    parser.add_argument("--mode", choices=["rule", "llm", "hybrid"], default="rule", help="Case cleaning strategy when raw input is used")
    parser.add_argument("--log", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log.upper())
    ingest(
        args.input,
        args.persist,
        args.graph,
        mode=args.mode,
        cleaned_jsonl=args.cleaned_jsonl,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
