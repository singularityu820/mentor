import pytest

from src.data.cleaner import (
    CaseEvent,
    CaseParticipant,
    CaseRecord,
    CleanedEvidence,
    RecordQuality,
)


@pytest.fixture
def sample_record() -> CaseRecord:
    return CaseRecord(
        case_id="CASE001",
        title="示例案件",
        court="某市中级人民法院",
        case_type="刑事判决书",
        charges=["故意伤害"],
        proceedings_summary="公诉机关指控被告人实施故意伤害行为。",
        factual_findings="经审理查明，被告人与被害人发生冲突并造成伤害。",
        judgment="判决如下：被告人判处有期徒刑三年。",
        participants=[CaseParticipant(name="李某", role="被告人")],
        timeline=[CaseEvent(timestamp="2021年1月1日", description="案发")],
        evidence=[
            CleanedEvidence(
                evidence_id="WITNESS_001",
                evidence_type="证人证言",
                summary="证人张某证明案发经过",
                source_excerpt="证人张某的证言……",
            )
        ],
        legal_basis=["《中华人民共和国刑法》第二百三十四条"],
        sentence_outcomes=["判决如下：被告人李某犯故意伤害罪，判处有期徒刑三年。"],
        raw_text="raw",
        quality=RecordQuality(
            completeness_score=0.8,
            missing_sections=[],
            evidence_coverage={"证人证言": 1},
            notes=[],
        ),
    )
