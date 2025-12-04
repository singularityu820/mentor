"""Knowledge retrieval utilities for augmenting agent reasoning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:  # pragma: no cover - optional dependency handling
    import chromadb  # type: ignore
except ImportError:  # pragma: no cover - gracefully degrade when Chroma is absent
    chromadb = None  # type: ignore

try:  # pragma: no cover - optional dependency handling
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover - gracefully degrade when NetworkX is absent
    nx = None  # type: ignore

from src.data.cleaner import CaseRecord

LOGGER = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Lightweight wrapper around the knowledge base assets.

    The retriever reads from two optional data sources:
    - A Chroma vector store (built via ``src.knowledge.importer``) for semantic search.
    - A node-link JSON graph describing entities and relationships.

    Both sources are optional; the retriever will simply return empty results if
    the assets are unavailable. This keeps the core runtime decoupled from the
    knowledge infrastructure while enabling richer prompts when data exists.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = Path("knowledge_store"),
        graph_path: Optional[Path] = Path("outputs/case_graph.json"),
        *,
        top_k: int = 3,
        entity_limit: int = 6,
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.graph_path = Path(graph_path) if graph_path else None
        self.top_k = top_k
        self.entity_limit = entity_limit
        self._collection = self._load_vector_collection()
        self._graph = self._load_graph()

    # ------------------------------------------------------------------
    def _load_vector_collection(self):
        if self.persist_dir is None or chromadb is None:
            if chromadb is None:
                LOGGER.debug("ChromaDB not installed; vector retrieval disabled")
            return None
        if not self.persist_dir.exists():
            LOGGER.debug("Knowledge store directory %s missing; skipping vector retrieval", self.persist_dir)
            return None
        try:
            if hasattr(chromadb, "PersistentClient"):
                client = chromadb.PersistentClient(path=str(self.persist_dir))  # type: ignore[attr-defined]
            else:  # pragma: no cover - legacy API
                from chromadb.config import Settings  # type: ignore

                client = chromadb.Client(  # type: ignore[misc]
                    Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(self.persist_dir))
                )
            return client.get_or_create_collection(name="crime_cases")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to initialise Chroma collection: %s", exc)
            return None

    def _load_graph(self):
        if self.graph_path is None or nx is None:
            if nx is None:
                LOGGER.debug("NetworkX not installed; graph reasoning disabled")
            return None
        if not self.graph_path.exists():
            LOGGER.debug("Knowledge graph file %s missing; skipping graph reasoning", self.graph_path)
            return None
        try:
            data = json.loads(self.graph_path.read_text(encoding="utf-8"))
            return nx.node_link_graph(data, edges="edges")  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load knowledge graph %s: %s", self.graph_path, exc)
            return None

    # ------------------------------------------------------------------
    def available(self) -> bool:
        return bool(self._collection or self._graph)

    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[str]:
        if not self._collection:
            return []
        if not query.strip():
            return []
        try:
            results = self._collection.query(  # type: ignore[attr-defined]
                query_texts=[query],
                n_results=top_k or self.top_k,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Vector retrieval failed: %s", exc)
            return []
        documents: List[str] = []
        for doc in results.get("documents", []):
            # results["documents"] is a list of lists
            documents.extend(doc)
        unique_docs = []
        seen = set()
        for doc in documents:
            snippet = (doc or "").strip()
            if not snippet:
                continue
            normalized = snippet[:500]
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_docs.append(normalized)
        return unique_docs[: (top_k or self.top_k)]

    def related_entities(self, case_id: str, limit: Optional[int] = None) -> List[str]:
        if not self._graph or not case_id:
            return []
        if case_id not in self._graph:
            return []
        entries: List[str] = []
        limit = limit or self.entity_limit
        for neighbor in self._graph.neighbors(case_id):  # type: ignore[call-arg]
            if len(entries) >= limit:
                break
            attrs = self._graph.nodes[neighbor]
            relation_lines = []
            if attrs.get("type") == "participant":
                relation_lines.append(f"参与人：{attrs.get('role', '-')}-{neighbor.split(':', 1)[-1]}")
            elif attrs.get("type") == "evidence":
                relation_lines.append(f"证据：{attrs.get('summary', '')[:120]}")
            else:
                relation_lines.append(f"关联节点：{neighbor}")
            entries.extend(relation_lines)
        return entries[:limit]

    # ------------------------------------------------------------------
    def build_context(
        self,
        *,
        query: str,
        case_record: Optional[CaseRecord] = None,
        top_k: Optional[int] = None,
        entity_limit: Optional[int] = None,
    ) -> Dict[str, str]:
        """Assemble human-readable snippets for agent prompts."""

        context: Dict[str, str] = {}
        documents = self.retrieve_documents(query, top_k=top_k)
        if documents:
            formatted = "\n".join(f"{idx + 1}. {doc}" for idx, doc in enumerate(documents))
            context["知识检索"] = formatted

        case_for_graph = case_record.case_id if case_record else ""
        if case_for_graph:
            related = self.related_entities(case_for_graph, limit=entity_limit)
            if related:
                context["图谱关联"] = "\n".join(related)
        return context