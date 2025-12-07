import os

from src.common import LLMClient


def test_llmclient_respects_crimentor_mock_env(monkeypatch):
    monkeypatch.setenv("CRIMEMENTOR_MOCK_LLM", "1")
    client = LLMClient()
    output = client.generate("system", "hello world")
    assert output.startswith("hello world")
    monkeypatch.delenv("CRIMEMENTOR_MOCK_LLM", raising=False)


def test_llmclient_respects_casesentinel_mock_env(monkeypatch):
    monkeypatch.setenv("CASESENTINEL_MOCK_LLM", "1")
    client = LLMClient()
    output = client.generate_chat([{"role": "user", "content": "foo"}])
    assert output.startswith("foo")
    monkeypatch.delenv("CASESENTINEL_MOCK_LLM", raising=False)
