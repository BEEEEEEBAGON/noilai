"""Probe/patching mechanics on a tiny random LLaMA with a character-level tokenizer."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from noilai.probe import extract, patching, probes

VOCAB_CHARS = list("abcdeghiklmnopqrstuvxyđ ăâêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ.,ÂÊÔ") + ["Ừ", "Ừ".lower()]


@pytest.fixture(scope="module")
def tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2}
    for ch in dict.fromkeys(VOCAB_CHARS):
        vocab.setdefault(ch, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", bos_token="<s>")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).eval()
    return model, fast


def test_examples_and_token_span(tiny):
    model, tok = tiny
    exs = extract.make_examples(["mèo", "bí"], encoding="nfc", carrier_ids=[0])
    assert len(exs) == 2 and exs[0].labels["tone"] == 1 and exs[1].labels["onset"] == "b"
    ex = exs[0]
    assert ex.text[ex.char_start:ex.char_end] == "mèo"
    enc = tok(ex.text, return_offsets_mapping=True, add_special_tokens=False)
    first, last = extract.token_span([tuple(o) for o in enc["offset_mapping"]], ex.char_start, ex.char_end)
    assert last - first + 1 == 3           # character tokenizer: one token per letter
    exs_nfd = extract.make_examples(["mèo"], encoding="nfd", carrier_ids=[0])
    assert len(exs_nfd[0].text[exs_nfd[0].char_start:exs_nfd[0].char_end]) == 4   # m e ` o


def test_hidden_state_extraction_shapes(tiny):
    model, tok = tiny
    exs = extract.make_examples(["mèo", "bí", "mật", "hoà"], encoding="nfc", carrier_ids=[0, 1])
    H = extract.extract_hidden_states(model, tok, exs, batch_size=3, add_special_tokens=False)
    assert set(H) == {"last", "after"}
    assert H["last"].shape == (8, 4, 32) and H["after"].shape == (8, 4, 32)     # 3 layers + embeddings


def test_probes_find_planted_linear_feature():
    rng = np.random.default_rng(0)
    n, L, d = 600, 3, 16
    tones = rng.integers(0, 6, size=n)
    groups = np.array([f"s{t}_{i % 40}" for i, t in enumerate(tones)])   # 240 syllable types
    H = rng.normal(size=(n, L, d))
    onehot = np.eye(6)[tones] * 4.0
    H[:, 1, :6] += onehot           # tone is linearly decodable at layer 1 only
    res = probes.run_layer_probes(H, tones, groups, feature="tone", seed=0)
    by_layer = {r.layer: r for r in res}
    assert by_layer[1].acc > 0.85 and by_layer[1].selectivity > 0.6
    assert by_layer[0].acc < 0.4 and by_layer[2].acc < 0.4
    assert probes.best_layer(res).layer == 1
    tab = probes.results_table(res)
    assert list(tab["layer"]) == [0, 1, 2]
    # the control task is fixed per group and follows the label distribution
    c = probes.control_labels(groups, tones, seed=1)
    assert len(np.unique(c)) <= 6 and all(len(set(c[groups == g])) == 1 for g in np.unique(groups)[:20])
    tr, te = probes.group_split(groups, 0.3, seed=0)
    assert not (set(groups[tr]) & set(groups[te]))


def test_patching_recovers_clean_answer_fully_at_full_patch(tiny):
    model, tok = tiny
    clean = tok("Chữ bí mật.", return_tensors="pt", add_special_tokens=False)["input_ids"]
    corrupt = tok("Chữ bị mật.", return_tensors="pt", add_special_tokens=False)["input_ids"]
    assert clean.shape == corrupt.shape
    diff_pos = (clean[0] != corrupt[0]).nonzero().flatten().tolist()
    assert len(diff_pos) == 1                                  # only the tone letter differs
    T = clean.shape[1]
    answer_pos = T - 1
    ta, tb = int(clean[0, diff_pos[0]]), int(corrupt[0, diff_pos[0]])
    all_pos = list(range(T))
    res = patching.run_patching(model, clean, corrupt, answer_pos, ta, tb,
                                position_groups=[diff_pos, all_pos], layers=None)
    assert res.recovery.shape == (3, 2)
    # patching every position at the last layer reproduces the clean logits exactly
    assert res.recovery[-1, 1] == pytest.approx(1.0, abs=1e-4)
    # patching only the differing position at the LAST layer cannot change the answer position's logits
    assert res.recovery[-1, 0] == pytest.approx(0.0, abs=1e-4)
    # patching the whole sequence at any layer also recovers the clean answer exactly
    assert np.allclose(res.recovery[:, 1], 1.0, atol=1e-4)


def test_steering_changes_logits(tiny):
    model, tok = tiny
    ids = tok("Chữ bí mật.", return_tensors="pt", add_special_tokens=False)["input_ids"]
    layers = patching.get_decoder_layers(model)
    assert len(layers) == 3
    H_a = np.random.default_rng(0).normal(size=(20, 32))
    H_b = np.random.default_rng(1).normal(size=(20, 32)) + 1.0
    d = patching.difference_in_means(H_a, H_b)
    assert np.linalg.norm(d) == pytest.approx(1.0)
    r0 = patching.steering_flip_rate(model, [ids], [ids.shape[1] - 1], 5, 6, layer_idx=1, direction=d, alpha=0.0)
    r1 = patching.steering_flip_rate(model, [ids], [ids.shape[1] - 1], 5, 6, layer_idx=1, direction=d, alpha=50.0)
    assert r0["mean_ld_shift"] == pytest.approx(0.0, abs=1e-5)
    assert abs(r1["mean_ld_shift"]) > 1e-3


def test_minimal_pairs_build_and_align(tiny):
    from noilai.probe import pairs as P
    from noilai.vi import lexicon as L
    model, tok = tiny
    inv = L.load_inventory()
    prs = P.build_pairs(inv, 30, seed=0)
    assert len(prs) == 30
    for pr in prs:
        assert pr.clean != pr.corrupt and pr.clean[pr.target_char_span[0]:pr.target_char_span[1]] == pr.target_syllable_clean
        assert pr.answer_clean != pr.answer_corrupt
    aligned = [P.align_pair(tok, pr, add_special_tokens=False) for pr in prs]
    ok = [a for a in aligned if a]
    assert len(ok) >= 20                      # character tokenizer: same length unless a letter count differs
    a = ok[0]
    assert len(a["clean_ids"]) == len(a["corrupt_ids"]) and a["tok_clean"] != a["tok_corrupt"]
    assert set(a["diff_positions"]) <= set(a["target_positions"])


def test_pairs_target_second_nfd_and_tone_only_readout(tiny):
    from noilai.probe import pairs as P
    from noilai.vi import lexicon as L
    from noilai.vi import unicode as U
    model, tok = tiny
    inv = L.load_inventory()
    prs = P.build_pairs(inv, 20, seed=1)                       # target second by default
    for pr in prs:
        words = pr.clean.split("→")[0].split()[-2:]
        assert words[1] == pr.target_syllable_clean and words[0] == pr.partner
        assert not pr.clean.endswith(" ")
    nfd = P.build_pairs(inv, 5, seed=1, encoding="nfd")
    for pr in nfd:
        s, e = pr.target_char_span
        assert pr.clean[s:e] == U.nfd(pr.target_syllable_clean)
        if U.nfd(pr.target_syllable_clean) != pr.target_syllable_clean:          # a target with diacritics
            assert U.encoding_form(pr.clean) in ("nfd", "mixed")
            assert U.nfc(pr.clean) != pr.clean and U.encoding_form(pr.clean[:s]) in ("nfc", "ascii")   # carrier stays NFC
    al = [P.align_pair(tok, pr, add_special_tokens=False) for pr in prs]
    ok = [a for a in al if a]
    assert ok and all("readout_tone_only" in a and len(a["readout_pieces"]) == 2 for a in ok)


def test_structural_baseline_matches_probe_when_label_is_a_token_function():
    rng = np.random.default_rng(3)
    n = 400
    tones = rng.integers(0, 6, size=n)
    token_ids = [[100 + t, 7] for t in tones]                # the token id encodes the tone exactly
    groups = [f"g{i}" for i in range(n)]
    coda = ["open"] * n
    base = probes.structural_baseline(token_ids, coda, tones, groups, seed=0)
    assert base > 0.95
    H = rng.normal(size=(n, 8))
    ex = probes.excess_over_structural(H, token_ids, coda, tones, groups, seed=0)
    assert ex["structural_baseline"] > 0.95 and ex["excess"] < 0
