from augur.tokenizer import Tokenizer


def test_tokenizer_exposes_eos_token_id() -> None:
    tokenizer = Tokenizer(
        vocab={"a": 0},
        merges=[],
        special_tokens={"<|endoftext|>": 3},
        eos_token="<|endoftext|>",
    )

    assert tokenizer.eos_token_id == 3
