import copy
import json
from pathlib import Path

from src.data.cleaner import (
    CaseRecord,
    clean_case_document,
    extract_case_id,
    extract_charges,
    extract_participants,
    normalize_whitespace,
)
from src.data.llm_cleaner import LLMCaseCleaner, _load_json_object


def test_normalize_whitespace_collapses_multiple_spaces():
    assert normalize_whitespace("  foo\n\nbar  ") == "foo bar"


def test_extract_case_id_handles_missing_pattern():
    text = "这是一份没有案号的判决书"
    assert extract_case_id(text) == "UNKNOWN_ID"


def test_extract_charges_handles_multiple_matches():
    text = "被告人张三犯故意杀人罪，另犯故意伤害罪。"
    charges = extract_charges(text)
    assert set(charges) == {"故意杀人", "故意伤害"}


def test_clean_case_document_parses_basic_fields():
    raw_entry = {
        "Case_info": (
            "山西省某某法院刑事判决书（2024）晋01刑初1号。"
            "公诉机关某市人民检察院。被告人李某犯故意杀人罪。"
            "经审理查明：2021年1月1日，被告人李某于案发地…本院认为："
            "被告人李某的行为构成故意杀人罪。判决如下：一、被告人李某犯故意杀人罪。"
        )
    }
    record: CaseRecord = clean_case_document(raw_entry)
    assert record.case_id == "（2024）晋01刑初1号"
    assert "经审理查明" in record.factual_findings
    assert record.charges[0] == "故意杀人"
    assert record.judgment.startswith("判决如下")


def test_extract_participants_identifies_roles():
    text = "被告人李某，辩护人张律师，公诉机关某市检察院。"
    participants = extract_participants(text)
    roles = {p.role for p in participants}
    assert {"被告人", "辩护人", "检察机关"}.issubset(roles)


class _StubLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate_chat(self, messages, **kwargs):
        if not self._responses:
            raise RuntimeError("No stub responses remaining")
        self.calls += 1
        return self._responses.pop(0)


def test_llm_case_cleaner_success(sample_record):
    payload = sample_record.model_dump(mode="python")
    payload.pop("quality", None)
    payload["case_id"] = "CASE_LLM"
    payload["participants"] = [
        {
            "name": "李某",
            "role": "被告人",
            "attributes": {},
        }
    ]
    payload["timeline"] = [
        {
            "timestamp": "2021年1月1日",
            "description": "案发",
            "source": "判决书",
        }
    ]
    payload["evidence"] = [
        {
            "evidence_id": "E001",
            "evidence_type": "书证",
            "summary": "证据摘要",
            "source_excerpt": "原文片段",
        }
    ]
    payload["raw_text"] = "原文"
    stub = _StubLLMClient([json.dumps(payload, ensure_ascii=False)])
    cleaner = LLMCaseCleaner(llm_client=stub, max_retries=0)
    record = cleaner.clean({"Case_info": sample_record.raw_text})
    assert record.case_id == "CASE_LLM"
    assert record.quality.completeness_score > 0
    assert stub.calls == 1


def test_llm_case_cleaner_falls_back(sample_record):
    stub = _StubLLMClient(["not-json"])
    fallback = copy.deepcopy(sample_record)
    cleaner = LLMCaseCleaner(llm_client=stub, max_retries=0)
    record = cleaner.clean({"Case_info": sample_record.raw_text}, fallback_record=fallback)
    assert record.case_id == sample_record.case_id
    assert stub.calls == 1


def test_load_json_object_tolerates_trailing_commas():
    raw = """
    {
        'items': [
            {'id': '1'},
        ],
        "note": "ok",
    }
    """
    loaded = _load_json_object(raw)
    assert loaded["items"][0]["id"] == "1"
    assert loaded["note"] == "ok"


def test_load_json_object_handles_fullwidth_punctuation():
    raw = """
    {
        "case_id"："ABC123",
        "charges"：[
            "故意杀人"
        ]，
        "note": "测试"
    }
    """
    loaded = _load_json_object(raw)
    assert loaded["case_id"] == "ABC123"
    assert loaded["charges"] == ["故意杀人"]


def test_load_json_object_escapes_inner_quotes():
    raw = """
    {
        "note": "接“110”报警",
        "desc": "使用‘单引号’作为引用"
    }
    """
    loaded = _load_json_object(raw)
    assert loaded["note"] in {'接"110"报警', '接“110”报警'}
    assert loaded["desc"] in {"使用'单引号'作为引用", "使用‘单引号’作为引用"}
