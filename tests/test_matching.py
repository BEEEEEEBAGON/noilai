"""noilai.stats.matching (PREREG 8.4 / 8.7 / 8.8): nearest-neighbour matching with exact strata and a caliper, the balance
diagnostics, the matched contrast with the base-pair bootstrap, the H3b tone-isolation DiD and the E2 item-audit join, on
synthetic data with planted structure."""
import json

import numpy as np
import pandas as pd
import pytest

from noilai.stats import e2
from noilai.stats import matching as M


def _pool(n, seed=0, shift=0.0, variants=("V1", "V3")):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"item_id": [f"c{i}" for i in range(n)], "variant": rng.choice(variants, n),
                         "x1": rng.normal(shift, 1, n), "x2": rng.normal(0, 2, n), "base_pair_id": [f"bp{i // 2}" for i in range(n)]})


def test_standardized_mean_difference_known_values_and_edge_cases():
    assert M.standardized_mean_difference([1, 2, 3], [1, 2, 3]) == 0.0
    assert M.standardized_mean_difference([1, 1], [2, 2]) == float("inf") and np.isnan(M.standardized_mean_difference([], [1]))
    d = M.standardized_mean_difference([0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0])     # means 0 and 0.5; variances 0 and 2/7 (ddof 1)
    assert d == pytest.approx(-0.5 / np.sqrt((0 + 2 / 7) / 2))
    # control weights shift the control mean (a control matched to two treated units counts twice)
    assert M.standardized_mean_difference([1.0, 1.0, 1.0], [0.0, 2.0], [1.0, 3.0]) < 0


def test_nearest_neighbour_match_respects_exact_strata_caliper_and_k_and_improves_balance():
    treated = pd.DataFrame({"item_id": [f"t{i}" for i in range(40)], "variant": ["V1"] * 20 + ["V3"] * 20,
                            "x1": np.linspace(-1, 1, 40), "x2": np.linspace(2, -2, 40)})
    pool = _pool(2000, shift=0.4)
    m = M.nearest_neighbour_match(treated, pool, exact=["variant"], nearest=["x1", "x2"], k_min=3, k_max=5, caliper_sd=0.2)
    assert m.n_treated == 40 and m.n_matched == 40 and not m.unmatched and 3 <= m.matches.groupby("treated_id").size().min()
    assert m.matches.groupby("treated_id").size().max() <= 5 and set(m.matches["rank"]) <= {1, 2, 3, 4, 5}
    # exact: every control shares its treated unit's variant
    pv = pool.set_index("item_id")["variant"]
    tv = treated.set_index("item_id")["variant"]
    assert all(pv[c] == tv[t] for t, c in zip(m.matches["treated_id"], m.matches["control_id"]))
    # the caliper: the RMS standardized (Mahalanobis) distance of every match is at most 0.2
    assert (m.matches["distance"] / np.sqrt(2) <= 0.2 + 1e-9).all()
    both = pd.concat([treated[["x1", "x2"]], pool[["x1", "x2"]]]).to_numpy()
    vinv = np.linalg.pinv(np.cov(both, rowvar=False))
    t1 = treated.set_index("item_id").loc[m.matches["treated_id"], ["x1", "x2"]].to_numpy()
    c1 = pool.set_index("item_id").loc[m.matches["control_id"], ["x1", "x2"]].to_numpy()
    dd = c1 - t1
    assert np.allclose(np.sqrt(np.einsum("ij,jk,ik->i", dd, vinv, dd)), m.matches["distance"])
    # one covariate: the caliper is |z| <= 0.2 pooled SD
    one = M.nearest_neighbour_match(treated, pool, exact=["variant"], nearest=["x1"], k_min=1, k_max=1, caliper_sd=0.2)
    sd1 = pd.concat([treated["x1"], pool["x1"]]).std(ddof=1)
    gap = pool.set_index("item_id").loc[one.matches["control_id"], "x1"].values - treated.set_index("item_id").loc[one.matches["treated_id"], "x1"].values
    assert np.all(np.abs(gap) / sd1 <= 0.2 + 1e-9) and one.matches.groupby("treated_id").size().max() == 1
    assert abs(m.smd_before["x1"]) > abs(m.smd_after["x1"]) and m.balanced is True and m.max_abs_smd_after < 0.1
    d = m.as_dict()
    assert d["n_pool"] == 2000 and d["mean_k"] == 5.0 and d["smd_balance_threshold"] == 0.1 and json.dumps(d)
    # ties and order are deterministic
    m2 = M.nearest_neighbour_match(treated, pool, exact=["variant"], nearest=["x1", "x2"], k_min=3, k_max=5, caliper_sd=0.2)
    assert m2.matches.equals(m.matches)
    # too tight a caliper or a missing stratum leaves treated units unmatched, counted by reason
    tight = M.nearest_neighbour_match(treated, pool.head(30), exact=["variant"], nearest=["x1", "x2"], caliper_sd=0.01)
    assert tight.n_matched < 40 and set(tight.unmatched) <= {"fewer_than_k_min_within_caliper", "no_control_in_exact_stratum"}
    none = M.nearest_neighbour_match(treated, pool.assign(variant="V2"), exact=["variant"], nearest=["x1", "x2"])
    assert none.n_matched == 0 and none.unmatched == {"no_control_in_exact_stratum": 40} and none.balanced is None
    with pytest.raises(KeyError):
        M.nearest_neighbour_match(treated, pool, exact=["variant"], nearest=["nope"])
    with pytest.raises(ValueError):
        M.nearest_neighbour_match(treated, pool, exact=["variant"], nearest=["x1"], k_min=0)


def test_matched_contrast_recovers_a_planted_effect_with_a_base_pair_bootstrap_ci():
    rng = np.random.default_rng(3)
    treated = pd.DataFrame({"item_id": [f"t{i}" for i in range(120)], "variant": "V1", "x1": rng.normal(0, 1, 120), "x2": rng.normal(0, 1, 120)})
    pool = _pool(3000, seed=4, variants=("V1",))
    m = M.nearest_neighbour_match(treated, pool, exact=["variant"], nearest=["x1", "x2"])
    assert m.n_matched >= 110
    # outcome: controls 50 %; treated 80 % (a planted +0.3)
    y_c = {c: float(rng.random() < 0.5) for c in pool["item_id"]}
    y_t = {t: float(rng.random() < 0.8) for t in treated["item_id"]}
    clusters = {t: f"abp{i // 2}" for i, t in enumerate(treated["item_id"])}
    mc = M.matched_contrast(y_t, y_c, m, clusters, n_boot=400, seed=0)
    assert 0.15 < mc.estimate < 0.45 and mc.lo < mc.estimate < mc.hi and mc.lo > 0 and mc.p_bootstrap < 0.01 and mc.p_sign_mcnemar < 0.01
    assert mc.n_treated == m.n_matched and mc.n_base_pairs == len({clusters[t] for t in m.controls_of()}) and mc.ci_method in ("bca", "percentile")
    assert set(mc.per_treated.columns) >= {"treated_id", "y_treated", "y_controls_mean", "k", "contribution", "cluster"}
    d = mc.as_dict()
    assert "ignores clustering" in d["p_sign_label"] and json.dumps(d)
    # a null effect: the interval covers 0
    y_t0 = {t: float(rng.random() < 0.5) for t in treated["item_id"]}
    mc0 = M.matched_contrast(y_t0, y_c, m, clusters, n_boot=400, seed=1)
    assert mc0.lo < 0 < mc0.hi
    with pytest.raises(ValueError):
        M.matched_contrast({}, y_c, m, clusters, n_boot=50)


def _audit_rows(items, iso_every=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, it in enumerate(items):
        nfc_total = int(rng.integers(2, 5))
        delta = int(rng.integers(1, 4))
        iso = i % iso_every == 0
        for enc, total in (("nfc", nfc_total), ("nfd", nfc_total + delta)):
            rows.append({"item_id": it, "tokenizer": "tok", "encoding": enc,
                         "input": {"tokens_mean": total / 2, "tokens_total": total, "align_among_split": 0.5 if total > 2 else None, "n_split": 2,
                                   "n_misaligned_split": 1 if iso else 0, "tone_isolated": iso and enc == "nfd", "byte_fallback": False,
                                   "any_single_token": total == 2, "all_single_token": False},
                         "delta_tokens_vs_nfc": delta if enc == "nfd" else 0})
    return rows


def test_read_items_audit_and_e2_frame_join_the_covariates_and_drop_items_without_an_audit_row(tmp_path):
    items = [f"T1-V1-{i:06d}" for i in range(50)]
    path = tmp_path / "items_tok__noilai_main.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _audit_rows(items)), encoding="utf-8")
    audit = M.read_items_audit(path)
    assert len(audit) == 100 and set(audit["encoding"]) == {"nfc", "nfd"} and audit["tokens_per_syllable"].notna().all()
    assert set(audit.columns) >= {"item_id", "tokenizer", "encoding", "tokens_per_syllable", "align_among_split", "n_split", "n_misaligned_split",
                                  "tone_isolated", "tokens_total", "delta_tokens_vs_nfc"}
    scores = pd.DataFrame({"item_id": [*items, "T1-V1-999999"], "model": "m", "correct": [i % 3 == 0 for i in range(51)],
                           "base_pair_id": [f"bp{i // 2}" for i in range(51)], "variant": "V1", "task": "T1", "input_lexical": True,
                           "output_lexical": False, "input_freq": [[5, 20]] * 51})
    f = M.e2_frame(scores, audit)
    assert len(f) == 50 and f["log_freq"].iloc[0] == pytest.approx(np.log1p(5)) and f["correct"].dtype.kind == "i"
    g = e2.prepare_covariates(f)
    assert {"split", "tps_w", "tps_b", "align_w"} <= set(g.columns)
    # the audit naming of scripts/kaggle_cpu_jobs.py is found by hf_id, family, model name, with and without the key
    cands = [p.name for p in M.items_audit_candidates(tmp_path, hf_id="org/tok", family="fam", key="noilai_main")]
    assert cands == ["items_org__tok__noilai_main.jsonl", "items_org__tok.jsonl", "items_fam__noilai_main.jsonl", "items_fam.jsonl"]
    assert M.find_items_audit(tmp_path, hf_id=None, family="tok", key="noilai_main") == path
    assert M.find_items_audit(tmp_path, hf_id=None, family="tok", key="noilai_core") is None
    # scripts/audit_items.py's flattener is the same function
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("audit_items", Path(__file__).resolve().parents[1] / "scripts" / "audit_items.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    row = _audit_rows(["x"])[0]
    assert mod.flatten_for_e2(row) == M.flatten_audit_row(row)


def test_h3b_did_recovers_a_planted_tone_isolation_effect_and_reports_power_and_balance():
    rng = np.random.default_rng(11)
    n = 600
    items = [f"T1-V3-{i:06d}" for i in range(n)]
    audit = pd.DataFrame([M.flatten_audit_row(r) for r in _audit_rows(items, iso_every=2, seed=1)])
    rows = []
    for i, it in enumerate(items):
        iso = i % 2 == 0
        base = rng.random() < 0.3
        # NFD helps the ISO items by +0.3, the NON-ISO items by 0
        nfd = (rng.random() < 0.6) if iso else (rng.random() < 0.3)
        common = {"item_id": it, "variant": "V3", "task": "T1", "base_pair_id": f"bp{i // 2}", "source": "lexicon" if i % 4 < 2 else "pseudo",
                  "input_freq": [int(rng.integers(0, 50)), int(rng.integers(0, 50))]}
        rows.append({**common, "arm": "nfc", "correct": base})
        rows.append({**common, "arm": "nfd", "correct": nfd})
    rows = pd.DataFrame(rows)
    r = M.h3b_did(rows, audit, variant="V3", n_boot=300, seed=0)
    assert r["computable"] and r["n_items"] == n and not r["underpowered"] and r["n_iso"] == 300 and r["n_non_iso"] == 300
    assert 0.15 < r["estimate"] < 0.55 and r["lo"] > 0 and r["p_bootstrap"] < 0.05
    m = r["matching"]
    assert m["n_matched"] >= 100 and m["k_min"] == m["k_max"] == 1 and m["exact"] == ["source"] and m["nearest"] == list(M.H3B_NEAREST)
    assert m["balanced"] and abs(m["smd_after"]["tokens_nfc"]) < 1e-9        # the token counts match exactly under the caliper
    assert r["effect_iso"] > r["effect_non_iso"] and "ISO" in r["estimand"] and json.dumps(r)
    # a cell below 500 items is flagged underpowered; a missing arm or variant is not computable
    small = M.h3b_did(rows[rows["item_id"].isin(items[:200])], audit, variant="V3", n_boot=100, seed=0)
    assert small["underpowered"] and small["n_items"] == 200
    assert not M.h3b_did(rows, audit, variant="V1", n_boot=50)["computable"]
    assert not M.h3b_did(rows[rows["arm"] == "nfc"], audit, variant="V3", n_boot=50)["computable"]
    assert not M.h3b_did(rows, audit[audit["encoding"] == "nfc"].assign(encoding="nfd").iloc[0:0], variant="V3", n_boot=50)["computable"]


def test_h6_covariates_and_tone_pair():
    cov = M.h6_covariates("mèo cái", "mái kèo", variant="V1", output_lexical=True, spelling_triggers=["c>k"], tone_pair="huyen-sac",
                          input_freq=[10, 3], output_freq=[0, 7])
    assert cov["has_spelling_trigger"] and cov["output_lexical"] and cov["task_class"] == "T1" and cov["variant"] == "V1"
    assert cov["log_freq_in_1"] == pytest.approx(np.log1p(10)) and cov["log_freq_out_1"] == 0.0 and cov["len_in_1"] == 3 and cov["len_out_2"] == 3
    assert set(cov) == set(M.H6_EXACT) | set(M.H6_NEAREST)
    assert M.tone_pair_of([{"tone": 1}, {"tone": 2}]) == "huyen-sac"
    with pytest.raises(ValueError):
        M.h6_covariates("một hai ba", "x y", variant="V1", output_lexical=True, spelling_triggers=[], tone_pair="a-b", input_freq=None, output_freq=None)
