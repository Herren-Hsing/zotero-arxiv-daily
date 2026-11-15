"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

from types import SimpleNamespace

import pytest

from tests.canned_responses import make_sample_paper, make_stub_openai_client


@pytest.fixture()
def llm_params():
    return {
        "api_mode": "chat_completion",
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api_mode", ["chat_completion", "response"])
def test_tldr_returns_response(llm_params, api_mode):
    llm_params["api_mode"] = api_mode
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "Hello! How can I assist you today?"
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_preserves_summary_instructions_with_long_full_text(llm_params, monkeypatch):
    import tiktoken

    client = make_stub_openai_client()
    create = client.chat.completions.create
    prompts = []

    def capture_prompt(**kwargs):
        prompts.append(kwargs["messages"][1]["content"])
        return create(**kwargs)

    monkeypatch.setattr(client.chat.completions, "create", capture_prompt)
    paper = make_sample_paper(full_text="word " * 10000)

    assert paper.generate_tldr(client, llm_params) == "Hello! How can I assist you today?"
    prompt, = prompts
    assert "one-paragraph TLDR summary" in prompt
    assert "problem and main results" in prompt
    assert "key methods and techniques" in prompt
    assert "specific contributions and innovations" in prompt
    assert len(tiktoken.encoding_for_model("gpt-4o").encode(prompt)) <= 4000


def test_response_mode_maps_max_tokens(llm_params):
    from types import SimpleNamespace

    received_kwargs = {}

    def create_response(**kwargs):
        received_kwargs.update(kwargs)
        return SimpleNamespace(output_text="Summary")

    client = SimpleNamespace(
        responses=SimpleNamespace(create=create_response),
    )
    llm_params["api_mode"] = "response"
    paper = make_sample_paper()

    assert paper.generate_tldr(client, llm_params) == "Summary"
    assert received_kwargs["max_output_tokens"] == 16384
    assert "max_tokens" not in received_kwargs


def test_invalid_api_mode_falls_back_to_abstract(llm_params):
    llm_params["api_mode"] = "invalid"
    paper = make_sample_paper()

    assert paper.generate_tldr(make_stub_openai_client(), llm_params) == paper.abstract


@pytest.mark.parametrize("api_mode", ["chat_completion", "response"])
def test_translate_abstract_uses_configured_model_and_source_text(llm_params, api_mode):
    llm_params["api_mode"] = api_mode
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output_text="中文译文。",
            choices=[SimpleNamespace(message=SimpleNamespace(content="中文译文。"))],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        responses=SimpleNamespace(create=create),
    )
    paper = make_sample_paper(abstract="We report a 15% improvement.")

    paper.translate_abstract(client, llm_params)

    call, = calls
    messages = call["input" if api_mode == "response" else "messages"]
    assert "Simplified Chinese" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "We report a 15% improvement."}
    assert call["model"] == llm_params["generation_kwargs"]["model"]
    assert paper.translated_abstract == "中文译文。"
    assert paper.abstract == "We report a 15% improvement."


def test_translate_abstract_skips_empty_input_and_leaves_original_on_error(llm_params):
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("translation unavailable")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    missing = make_sample_paper(abstract=" ")
    available = make_sample_paper(abstract="The original abstract")

    missing.translate_abstract(client, llm_params)
    available.translate_abstract(client, llm_params)

    assert len(calls) == 1
    assert missing.translated_abstract is None
    assert available.translated_abstract is None
    assert available.abstract == "The original abstract"


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api_mode", ["chat_completion", "response"])
def test_affiliations_returns_parsed_list(llm_params, api_mode):
    llm_params["api_mode"] = api_mode
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # re.search for [...] will fail -> AttributeError -> caught -> returns None
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None
