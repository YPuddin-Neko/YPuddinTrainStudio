import pytest
import torch

pytest.importorskip("transformers")

from ypuddin.models.anima.text import ASSETS, PAD_FLOOR, AnimaText  # noqa: E402

pytestmark = pytest.mark.skipif(not (ASSETS / "qwen3_06b").exists(), reason="anima assets not vendored")


def test_tokenizers_and_padding_conventions():
    tp = AnimaText(ASSETS / "qwen3_06b")  # tokenizer-only; encoder weights never loaded
    q_ids, q_mask, t5_ids, t5_mask = tp._tokenize(["1girl, smile", ""])
    assert q_ids.shape[0] == 2 and t5_ids.shape[0] == 2
    assert t5_ids[0, int(t5_mask[0].sum()) - 1].item() == 1  # T5 EOS terminates the sequence
    assert t5_ids[1].tolist()[0] == 1  # empty caption -> just </s>
    entries = [{"embeds": torch.randn(7, 1024), "t5_ids": t5_ids[0][: int(t5_mask[0].sum())]}, {"embeds": torch.randn(3, 1024), "t5_ids": torch.tensor([1])}]
    cond = tp.cond_from_cache(entries, "cpu")
    assert cond["embeds"].shape == (2, PAD_FLOOR, 1024)
    assert cond["attn_mask"].sum(1).tolist() == [7, 3]
    assert cond["t5_ids"].shape == (2, PAD_FLOOR) and cond["t5_mask"].sum(1).tolist()[1] == 1
    assert torch.all(cond["embeds"][0, 7:] == 0)  # padded positions are zero vectors
