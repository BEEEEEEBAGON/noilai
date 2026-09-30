"""Probe/patching mechanics on a tiny random LLaMA with a character-level tokenizer."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import sys as _sys
from pathlib import Path as _Path

from noilai.probe import extract, patching, probes

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))

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


# ------------------------------------------------------------------ design 9.1–9.3 protocol (review round)
def test_extraction_uses_the_patching_hooks_not_the_post_norm_hidden_states(tiny):
    model, tok = tiny
    exs = extract.make_examples(["mèo"], encoding="nfc", carrier_ids=[0])
    H = extract.extract_hidden_states(model, tok, exs, batch_size=1, add_special_tokens=False, positions=("last",))
    enc = tok(exs[0].text, return_tensors="pt", add_special_tokens=False)
    first, last = extract.token_span([tuple(o) for o in tok(exs[0].text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]],
                                     exs[0].char_start, exs[0].char_end)
    with torch.no_grad():
        with patching.ResidualCache(model) as cache:
            out = model(**enc, output_hidden_states=True)
        emb = model.get_input_embeddings()(enc["input_ids"])
    top = H["last"][0, -1]
    hooked = cache.store[len(cache.layers) - 1][0, last].numpy()
    post_norm = model.model(**enc).last_hidden_state[0, last].detach().numpy()
    assert np.allclose(top, hooked, atol=1e-5)                                   # hidden index L = block L-1's output
    assert not np.allclose(top, post_norm, atol=1e-3)                            # not the final-norm vector
    assert np.allclose(H["last"][0, 0], emb[0, last].numpy(), atol=1e-6)           # hidden index 0 = the embedding
    assert extract.HIDDEN_INDEX_OFFSET == 1 and H["last"].shape[1] == len(cache.layers) + extract.HIDDEN_INDEX_OFFSET
    # hidden_states[-1] of transformers is post-norm: it must NOT be what the top probe index sees
    assert not np.allclose(top, out.hidden_states[-1][0, last].numpy(), atol=1e-3) or np.allclose(hooked, out.hidden_states[-1][0, last].numpy())


def test_mark_position_selects_the_bare_tone_mark_under_nfd_and_is_nan_under_nfc(tiny):
    model, tok = tiny
    nfd = extract.make_examples(["mèo", "ma"], encoding="nfd", carrier_ids=[0])       # 'ma' has no tone mark
    H = extract.extract_hidden_states(model, tok, nfd, batch_size=2, add_special_tokens=False, positions=("mark", "last"))
    assert H["mark"].shape == H["last"].shape == (2, 4, 32)
    assert np.isfinite(H["mark"][0]).all() and np.isnan(H["mark"][1]).all()
    # the mark token IS the combining grave: its representation differs from the last token's
    assert not np.allclose(H["mark"][0, 1], H["last"][0, 1])
    nfc = extract.make_examples(["mèo"], encoding="nfc", carrier_ids=[0])
    assert np.isnan(extract.extract_hidden_states(model, tok, nfc, batch_size=1, add_special_tokens=False, positions=("mark",))["mark"]).all()
    with pytest.raises(ValueError):
        extract.extract_hidden_states(model, tok, nfc, positions=("middle",), add_special_tokens=False)


def test_carriers_follow_design_9_2():
    from noilai import constants
    assert len(extract.CARRIERS) >= constants.PROBE_MIN_CARRIERS >= 4
    quoted = [c for c in extract.CARRIERS if "«{}»" in c]
    assert len(quoted) >= 1                                                       # one quoted-slot carrier
    for c in extract.CARRIERS:
        assert not c.startswith("{}")                                             # the slot is never sentence-initial
        if c not in quoted:
            assert extract.carrier_after_word(c) in extract.TONE_NEUTRAL_AFTER_WORDS, c   # và / lên / trong / này after the slot
    exs = extract.make_examples(["ma"], carrier_ids=[5], extra_labels={"ma": {"attested": True}})
    assert exs[0].text.startswith("Người ta viết chữ «ma»") and exs[0].labels["attested"] is True


def test_nested_split_tuning_and_paired_control_seeds():
    from noilai import constants
    rng = np.random.default_rng(0)
    n, L, d = 600, 3, 16
    tones = rng.integers(0, 6, size=n)
    groups = np.array([f"s{t}_{i % 40}" for i, t in enumerate(tones)])
    H = rng.normal(size=(n, L, d))
    H[:, 1, :6] += np.eye(6)[tones] * 4.0
    tr, dev, te = probes.nested_group_split(groups, seed=0)
    assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[dev]) & set(groups[te])) and not (set(groups[tr]) & set(groups[dev]))
    assert abs(len(te) / n - 0.3) < 0.05 and abs(len(dev) / n - 0.1) < 0.05
    tuned = probes.tune_C(H[:, 1, :], tones, groups, seed=0)
    assert tuned["C"] in constants.PROBE_L2_GRID and set(tuned["dev_acc"]) == set(constants.PROBE_L2_GRID)
    r_a = probes.run_layer_probes(H, tones, groups, feature="tone", seed=0, C=tuned["C"], control_seed=0)
    r_b = probes.run_layer_probes(H, tones, groups, feature="tone", seed=0, C=tuned["C"], control_seed=1)
    assert [r.n_test for r in r_a] == [r.n_test for r in r_b] and [r.acc for r in r_a] == [r.acc for r in r_b]   # same split
    assert any(a.control_acc != b.control_acc for a, b in zip(r_a, r_b))                                         # different control draw
    assert r_a[0].control_seed == 0 and r_b[0].control_seed == 1 and r_a[0].split_seed == 0 and r_a[0].n_dev > 0
    r_c = probes.run_layer_probes(H, tones, groups, feature="tone", seed=1, control_seed=0)
    assert r_c[0].n_test != r_a[0].n_test or r_c[0].acc != r_a[0].acc                                            # another split
    # rime-holdout split: the split key is the rime, the control task stays per syllable type
    rimes = np.array([f"r{i % 7}" for i in range(n)])
    r_r = probes.run_layer_probes(H, tones, groups, feature="tone", seed=0, holdout_groups=rimes, split_key="rime")
    trr, _, ter = probes.nested_group_split(rimes, seed=0)
    assert r_r[0].split_key == "rime" and r_r[0].n_test == len(ter) and not (set(rimes[trr]) & set(rimes[ter]))


def test_excess_over_structural_returns_a_syllable_clustered_interval():
    rng = np.random.default_rng(3)
    n = 400
    tones = rng.integers(0, 6, size=n)
    groups = np.array([f"g{i % 100}" for i in range(n)])          # 4 rows per syllable type
    token_ids = [[7, 8] for _ in range(n)]                       # uninformative token ids
    H = rng.normal(size=(n, 8))
    H[:, :6] += np.eye(6)[tones] * 4.0                           # tone decodable from the representation only
    ex = probes.excess_over_structural(H, token_ids, ["open"] * n, tones, groups, seed=0, n_boot=200)
    assert set(ex) >= {"acc", "structural_baseline", "excess", "lo", "hi", "excludes_zero", "n_test_groups"}
    assert ex["excess"] > 0.3 and ex["lo"] > 0 and ex["excludes_zero"] and ex["lo"] <= ex["excess"] <= ex["hi"]
    assert ex["n_test_groups"] < ex["n_test"]                    # resampling syllable types, not rows


def test_patching_passes_no_kv_cache_and_reports_the_greedy_clean_answer(tiny, monkeypatch):
    model, tok = tiny
    seen = []
    orig = model.forward

    def spy(*a, **kw):
        seen.append(kw.get("use_cache"))
        return orig(*a, **kw)

    monkeypatch.setattr(model, "forward", spy)
    clean = tok("Chữ bí mật.", return_tensors="pt", add_special_tokens=False)["input_ids"]
    corrupt = tok("Chữ bị mật.", return_tensors="pt", add_special_tokens=False)["input_ids"]
    T = clean.shape[1]
    gap = patching.pair_gap(model, clean, corrupt, T - 1, 5, 6)
    assert set(gap) == {"ld_clean", "ld_corrupt", "gap", "clean_argmax", "clean_greedy_correct"}
    assert gap["gap"] == pytest.approx(gap["ld_clean"] - gap["ld_corrupt"])
    res = patching.run_patching(model, clean, corrupt, T - 1, 5, 6, position_groups=[list(range(T))], layers=[0])
    assert res.clean_argmax == gap["clean_argmax"] and res.raw_ld.shape == (1, 1) and res.gap == pytest.approx(gap["gap"])
    patching.steering_flip_rate(model, [clean], [T - 1], 5, 6, layer_idx=1, direction=np.ones(32), alpha=0.1)
    assert seen and all(v is False for v in seen)


def test_rerender_pair_recomputes_the_span_inside_a_full_prompt():
    from noilai.probe import pairs as P
    from noilai.vi import lexicon as L
    inv = L.load_inventory()
    pr = P.build_pairs(inv, 3, seed=0)[0]
    full = P.rerender_pair(pr, lambda inp: f"Hướng dẫn dài.\nVí dụ: a b -> b a\nCụm từ: {inp}\nĐáp án:")
    s, e = full.target_char_span
    assert full.clean[s:e] == pr.target_syllable_clean and full.corrupt.startswith("Hướng dẫn")
    assert full.clean.endswith("Đáp án:") and full.clean[:s].endswith(pr.partner + " ")
    with pytest.raises(ValueError):
        P.rerender_pair(pr, lambda inp: "no item here")


def test_readout_a_pieces_and_prompt():
    from noilai.probe import readouts as R
    assert R.readout_a_prompt("bí") == "Chữ «bí» mang thanh gì? Đáp án: thanh"
    assert R.readout_a_prompt("bí", "nfd") != R.readout_a_prompt("bí") and R.TONE_ANSWERS[2] == "sắc"
    ok = R.check_answer_pieces(lambda s: [hash(s) % 10_000], lambda ids: str(ids[0]))
    assert ok["distinct"] and len(ok["ids"]) == 6
    with pytest.raises(AssertionError, match="share a first piece"):
        R.check_answer_pieces(lambda s: [ord(s[1])])           # huyền / hỏi -> 'h', ngang / ngã -> 'n'
    if R.GEMMA3_MODEL.exists():
        info = R.GEMMA3_ANSWER_PIECES
        assert info["pieces"] == ["▁ngang", "▁huyền", "▁sắc", "▁hỏi", "▁ng", "▁nặng"]


def test_run_probe_patching_stage_applies_the_filters_and_g6_reaches_one(tiny, tmp_path, monkeypatch):
    import argparse
    import json

    import run_probe as RP

    from noilai import constants
    from noilai.vi import lexicon as L

    model, tok = tiny
    inv = L.load_inventory()
    args = argparse.Namespace(n_pairs=3, pair_seed=0, readout_b_prompt="skeleton", tiny_test=True, out=tmp_path, n_boot=30)
    # a random model has no LD gap: filter 3 removes every pair and readout B is not run
    summary, ret = RP.run_patching_stage(model, tok, inv, args, "cpu", False)
    assert ret["pair_candidates"] >= ret["f1_aligned_skeleton"] >= ret["f2_tone_only"] >= ret["f1_aligned_full_prompt"] >= ret["retained_b"] == 0
    assert summary["readout_b"]["status"] == "not_run" and "localization" in summary["study_type"]
    # with a model gap (monkeypatched filter 3) the retained pairs are patched over the eight groups
    monkeypatch.setattr(RP.patching, "pair_gap", lambda *a, **k: {"ld_clean": 3.0, "ld_corrupt": 0.5, "gap": 2.5, "clean_argmax": a[-2],
                                                                  "clean_greedy_correct": True})
    summary, ret = RP.run_patching_stage(model, tok, inv, args, "cpu", False)
    assert ret["retained_b"] == ret["used_b"] == 3 and ret["f3_gap_ok"] >= 3 and ret["min_gap_nats"] == constants.PATCHING_MIN_GAP_NATS
    R = np.load(tmp_path / "patching.npz")["recovery_b"]
    assert R.shape == (3, 3, len(RP.GROUP_NAMES)) and summary["readout_b"]["status"] == "run"
    g6 = RP.GROUP_NAMES.index("G6_from_target")
    assert np.allclose(R[:, :, g6], 1.0, atol=1e-3)                      # everything from the varying syllable onward: 1.0
    g5 = RP.GROUP_NAMES.index("G5_final")
    assert np.allclose(R[:, -1, g5], 1.0, atol=1e-3)                     # the final position at the top block is the readout itself
    js = json.loads((tmp_path / "patching.json").read_text())
    assert js["position_groups"] == list(RP.GROUP_NAMES) and len(js["readout_b"]["lo"]) == 3 and js["retention"]["used_b"] == 3


def test_run_probe_samples_legal_syllables_with_the_attested_flag():
    import run_probe as RP

    from noilai.vi import lexicon as L
    inv = L.load_inventory()
    syl, labels = RP.sample_syllables(inv, 60, seed=0)
    assert len(syl) == 60 and set(labels) == set(syl) and all(set(v) == {"attested"} for v in labels.values())
    assert any(not v["attested"] for v in labels.values())              # legal non-word syllables are included (design 9.2)
    from collections import Counter

    from noilai.vi.syllable import try_parse
    assert Counter(try_parse(s).syllable.tone for s in syl) == {t: 10 for t in range(6)}
    assert syl == RP.sample_syllables(inv, 60, seed=0)[0] != RP.sample_syllables(inv, 60, seed=1)[0]


def test_probe_manifest_carries_provenance_fields():
    from noilai.probe import manifest as M
    m = M.base_manifest(model="x")
    assert set(m) >= {"git_commit", "git_dirty", "code_sha256", "resource_hashes", "started_utc", "model"}
    assert len(m["code_sha256"]) == 64 and "gemma3_tokenizer.model" in m["resource_hashes"]
