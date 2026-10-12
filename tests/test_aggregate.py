"""noilai.stats.aggregate: from scored run directories to the cells of the pre-registered analysis, on synthetic
rows written to run directories under tmp (and on the local echo smoke run when it exists). Covered: the run-id /
prompt-id parsing and the line pooling (p0p1 + p2 pooled, cpu apart, tpu_main -> E1_main, duplicates refused),
paraphrase means with known values and the mixed-cell refusal (item 52), every PREREG 8.2 cell field with the
small-cell flags forwarded from noilai.stats.bootstrap, the equal-weight primary metric with its p0 / range and the
API variant, the all-three-paraphrases rule, the within-item ATE tau_m(a) with its base-pair bootstrap CI around
the planted truth, the census "0 by construction" and refused rows, the explicit-input gap, the T3 headline with
the log-probability modes kept apart, the ledger's done-only selection, and the writer's refusals."""
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from noilai import constants
from noilai.stats import aggregate as A

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
SB = importlib.import_module("_smoke_build")       # tests/_smoke_build.py: the shared smoke-build locator
# the scored echo smoke run of the session that wrote these tests (20 T1 rows, all unparseable); skipped when absent
ECHO_RUNS = SB.smoke_build_dir().parent.parent / "runs_echo"
N_BOOT = 200
P0, P1, P2 = ("vi-p0-s3-explained-raw", "vi-p1-s3-explained-raw", "vi-p2-s3-explained-raw")
COMP0 = "vi-p0-s3-explained-components"
VARIANTS = ("V1", "V2", "V3", "V4")


# ------------------------------------------------------------------ synthetic items, outcomes and run directories
class Sim:
    """Probit-threshold outcomes with a base-pair random effect: P(Y = 1) is exactly `p`, items of one base pair
    are correlated (ICC `icc`), and two conditions on the same item differ by exactly p_b - p_a in expectation."""

    def __init__(self, seed: int, icc: float = 0.3):
        self.rng = np.random.default_rng(seed)
        self.icc = icc
        self.u: dict[str, float] = {}

    def y(self, bp: str, p: float) -> bool:
        u = self.u.setdefault(bp, float(self.rng.normal()))
        return bool(np.sqrt(self.icc) * u + np.sqrt(1 - self.icc) * self.rng.normal() < norm.ppf(p))


def t1_items(n_pairs: int, variants=VARIANTS, task="T1", core_every=2, c2_every=3) -> list[dict]:
    items = []
    for bp in range(n_pairs):
        for v in variants:
            items.append({"item_id": f"{task}-{v}-{bp:06d}", "task": task, "variant": v, "base_pair_id": f"bp-{bp:05d}",
                          "source": "lexicon" if bp % 2 else "pseudo", "in_core": bp % core_every == 0,
                          "c2_affected": bp % c2_every == 0, "gold": ["x y"], "pair_item_id": None, "twin_type": None})
    return items


def t3_items(n_pairs: int, variants=("V1", "V2")) -> list[dict]:
    """Yes / no pairs; every fourth `no` twin is a spelling twin."""
    items = []
    for bp in range(n_pairs):
        for v in variants:
            yes, no = f"T3-{v}-{2 * bp:06d}", f"T3-{v}-{2 * bp + 1:06d}"
            common = {"task": "T3", "variant": v, "base_pair_id": f"bp3-{bp:05d}", "source": "lexicon" if bp % 2 else "pseudo",
                      "in_core": bp % 2 == 0, "c2_affected": bp % 3 == 0}
            items.append({**common, "item_id": yes, "gold": "yes", "pair_item_id": no, "twin_type": None})
            items.append({**common, "item_id": no, "gold": "no", "pair_item_id": yes,
                          "twin_type": "spelling" if bp % 4 == 0 else "onset"})
    return items


def xcopa_items(n: int) -> list[dict]:
    return [{"item_id": f"xcopa-vi-test-{i:04d}", "task": "XCOPA", "variant": "", "gold": "1"} for i in range(n)]


def row(item: dict, arm: str, prompt_id: str, correct: bool, **extra) -> dict:
    """A scores.jsonl row as scripts/score_run.py writes it (no model column; base_pair_id / source copied from the item)."""
    r = {k: v for k, v in item.items() if k != "gold"}
    r.update({"gold": item.get("gold"), "arm": arm, "arm_scope": "whole_prompt", "prompt_id": prompt_id, "correct": bool(correct),
              "correct_lenient": bool(correct), "error_class": "correct" if correct else "wrong", "answer": "x"})
    r.update(extra)
    return r


def manifest(run_id: str, model: str, item_path: Path, *, status="finished", census=None, smoke=False, backend="vllm",
             n_thinking=0) -> dict:
    return {"run_id": run_id, "status": status, "n_thinking_chars_total": n_thinking,
            "identity": {"run_id": run_id, "model_key": model, "model_id": None, "revision": None},
            "model": {"name": model, "provider": None, "hardware_entry": "api" if backend == "openai_compat" else "t4",
                      "config_entry": {"name": model, "backend": backend}},
            "options": {"notes": {"smoke": smoke}, "arms": ["nfc"]},
            "item_file": {"path": str(item_path), "sha256": "0" * 64, "kind": "noilai"},
            "data": {"item_file": str(item_path), "item_file_sha256": "0" * 64,
                     "normalization_census": census or {"status": "not_censused", "classification": None}}}


def write_run(root: Path, run_id: str, model: str, rows: list[dict], item_name="noilai_main.jsonl", **kw) -> Path:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest(run_id, model, root / "items" / item_name, **kw)), encoding="utf-8")
    (d / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return d


def censused(nfd="passes_through", pc="passes_through") -> dict:
    return {"status": "censused", "source": "data/audit/synthetic.json", "classification": nfd, "legacy_census": False,
            "verdict_nfd": nfd, "verdict_pc": pc, "engine": "vllm"}


# ------------------------------------------------------------------ names
def test_run_id_prompt_id_and_item_file_parsing():
    assert A.split_run_id("E1_main__open-a--thinking__p2") == ("E1_main", "open-a--thinking", "p2")
    assert A.split_run_id("E3_noilai__open-a") == ("E3_noilai", "open-a", "")
    assert A.split_run_id("interim_test") == ("interim_test", None, "")
    # the paraphrase split pools, the CPU engine stays apart, tpu_main is E1_main's TPU part
    assert A.pool_key("E1_main", "m", "p0p1") == A.pool_key("tpu_main", "m", "p2") == ("E1_main", "m", "")
    assert A.pool_key("E1_main", "m", "cpu") == ("E1_main", "m", "cpu") != A.pool_key("E1_main", "m", "")
    assert A.paraphrase_of(P1) == "p1" and A.paraphrase_of("xcopa-p2") == "p2" and A.paraphrase_of("odd") is None
    assert A.condition_of(P0) == A.condition_of(P2) == A.DEFAULT_CONDITION == "vi-*-s3-explained-raw"
    assert A.condition_of(COMP0) == A.COMPONENTS_CONDITION and A.condition_of(P0 + "-item") == "vi-*-s3-explained-raw-item"
    assert A.item_file_kind("data/release/v0.3/noilai_c2.jsonl") == "c2" and A.item_file_kind("noilai_main.jsonl") == "main"
    assert A.item_file_kind(None) == "main" and A.BASELINE_ARM == "nfc" and A.CLUSTER == constants.BOOTSTRAP_CLUSTER
    assert A.normalizing_arms(censused("normalizes", "normalizes")) == {"nfd", "win1258"}
    assert A.corrupting_arms(censused("corrupts")) == {"nfd"} and A.normalizing_arms(None) == set()


# ------------------------------------------------------------------ loading and pooling
def _e1_rows(items, sim, p, paraphrases=(P0, P1, P2)) -> list[dict]:
    return [row(it, "nfc", pid, sim.y(it["base_pair_id"], p)) for it in items for pid in paraphrases]


def test_load_runs_pools_the_paraphrase_split_keeps_cpu_apart_and_refuses_duplicates(tmp_path):
    items = t1_items(30, variants=("V1",))
    sim = Sim(1)
    write_run(tmp_path, "E1_main__open-a__p0p1", "open-a", _e1_rows(items, sim, 0.6, (P0, P1)))
    write_run(tmp_path, "E1_main__open-a__p2", "open-a", _e1_rows(items, sim, 0.6, (P2,)))
    write_run(tmp_path, "E1_main__open-a__cpu", "open-a", _e1_rows(items, sim, 0.6, (P0,)))
    write_run(tmp_path, "tpu_main__open-b", "open-b", _e1_rows(items, sim, 0.4, (P0,)))
    write_run(tmp_path, "E1_main__open-c", "open-c", _e1_rows(items, sim, 0.5, (P0,)), status="running")
    write_run(tmp_path, "E1_main__open-d", "open-d", _e1_rows(items, sim, 0.5, (P0,)), status="finished_thinking_present", n_thinking=12)
    write_run(tmp_path, "smoke_20__open-e", "open-e", _e1_rows(items, sim, 0.5, (P0,)), smoke=True)
    (tmp_path / "not_a_run").mkdir()
    L = A.load_runs(tmp_path)
    df = L.rows
    assert {r.run_id for r in L.runs} == {"E1_main__open-a__p0p1", "E1_main__open-a__p2", "E1_main__open-a__cpu", "tpu_main__open-b"}
    reasons = {Path(s["run_dir"]).name: s["reason"] for s in L.skipped}
    assert "running" in reasons["E1_main__open-c"] and "thinking" in reasons["E1_main__open-d"] and "smoke" in reasons["smoke_20__open-e"]
    # the model comes from the manifest, the line is aliased, the tag is kept per row, paraphrase / condition parsed
    assert set(df["model"]) == {"open-a", "open-b"} and set(df["line"]) == {"E1_main"}
    a = df[df["model"] == "open-a"]
    assert set(a["tag"]) == {"p0p1", "p2", "cpu"} and set(a["paraphrase"]) == {"p0", "p1", "p2"}
    assert (a["condition"] == A.DEFAULT_CONDITION).all() and (df["item_file"] == "main").all() and not df["is_api"].any()
    assert len(a[a["tag"] != "cpu"]) == 90 and len(a[a["tag"] == "cpu"]) == 30      # p0p1 + p2 pooled; cpu a unit apart
    assert L.n()["items"] == 30 and L.n()["base_pairs"] == 30 and L.n()["models"] == 2 and L.n()["runs"] == 4
    assert L.runs_table.loc[L.runs_table["run_id"] == "tpu_main__open-b", "line"].iloc[0] == "E1_main"
    assert L.census["open-a"]["status"] == "not_censused" and df["census_nfd"].isna().all()
    # the same paraphrase scored twice for one line is never averaged away
    write_run(tmp_path, "E1_main__open-a__p2", "open-a", _e1_rows(items, sim, 0.6, (P1, P2)))
    with pytest.raises(ValueError, match="E1_main__open-a__p0p1"):
        A.load_runs(tmp_path)
    # the smoke run can be asked for; explicit run directories and manifest paths are accepted too
    L2 = A.load_runs([tmp_path / "smoke_20__open-e", tmp_path / "tpu_main__open-b" / "manifest.json"], include_smoke=True)
    assert {r.model for r in L2.runs} == {"open-e", "open-b"}


def test_attach_clusters_from_the_item_file_and_item_id_for_xcopa(tmp_path):
    items = t1_items(5, variants=("V1",))
    item_path = tmp_path / "items.jsonl"
    item_path.write_text("".join(json.dumps(it) + "\n" for it in items), encoding="utf-8")
    rows = [row(it, "nfc", P0, True) for it in items]
    for r in rows:
        del r["base_pair_id"], r["source"]
    man = {"item_file": {"path": str(item_path)}, "data": {}}
    note = A.attach_clusters(rows, man)
    assert "5 rows took base_pair_id" in note and all(r["base_pair_id"] == f"bp-{i:05d}" for i, r in enumerate(rows))
    assert rows[1]["source"] == "lexicon" and rows[0]["source"] == "pseudo"
    xrows = [row(it, "nfd", "xcopa-p0", True, logprob_correct=True) for it in xcopa_items(3)]
    assert "cluster by item_id" in A.attach_clusters(xrows, man)
    assert [r["base_pair_id"] for r in xrows] == [r["item_id"] for r in xrows] and {r["source"] for r in xrows} == {A.NO_STRATUM}
    gone = {"item_file": {"path": str(tmp_path / "nowhere.jsonl")}, "data": {}}
    with pytest.raises(A.Unscorable, match="not on this machine"):
        A.attach_clusters([dict(r, base_pair_id=None) for r in rows], gone)


def test_ledger_mode_selects_done_units_and_their_arms_and_skips_unverified_hashes(tmp_path):
    items = t1_items(20, variants=("V1",))
    sim = Sim(2)
    rows = [row(it, arm, P0, sim.y(it["base_pair_id"], 0.5)) for it in items for arm in ("nfc", "nfd", "win1258")]
    write_run(tmp_path, "E3_noilai__open-a", "open-a", rows)
    write_run(tmp_path, "E3_noilai__open-b", "open-b", rows)
    write_run(tmp_path, "E3_noilai__open-c", "open-c", rows)
    ledger = tmp_path / "ledger.csv"
    cols = ["unit_id", "run_id", "model", "arm", "tag", "status", "hash_verified", "result_file", "experiment"]
    units = [("E3_noilai__open-a__nfc", "E3_noilai", "open-a", "nfc", "", "done", "yes", "", "E3"),
             ("E3_noilai__open-a__pc", "E3_noilai", "open-a", "pc", "", "done", "yes", "", "E3"),     # plan spelling pc -> win1258
             ("E3_noilai__open-a__nfd", "E3_noilai", "open-a", "nfd", "", "planned", "no", "", "E3"),
             ("E3_noilai__open-b__nfc", "E3_noilai", "open-b", "nfc", "", "done", "no", "", "E3"),
             ("E3_noilai__open-z__nfc", "E3_noilai", "open-z", "nfc", "", "done", "yes", "", "E3")]
    with open(ledger, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(units)
    L = A.load_runs(tmp_path, ledger=ledger)
    assert {r.run_id for r in L.runs} == {"E3_noilai__open-a"} and set(L.rows["arm"]) == {"nfc", "win1258"}
    assert L.runs[0].unit_ids == ["E3_noilai__open-a__nfc", "E3_noilai__open-a__pc"] and L.runs[0].hash_verified == "yes"
    reasons = {s["unit_id"]: s["reason"] for s in L.skipped}
    assert "hash does not match" in reasons["E3_noilai__open-b__nfc"] and "no run directory" in reasons["E3_noilai__open-z__nfc"]


# ------------------------------------------------------------------ paraphrase means and cells
def _frame(rows: list[dict], model="open-a", line="E1_main") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["model"], df["line"], df["tag"], df["item_file"], df["is_api"] = model, line, "", "main", False
    df["c2_enriched"], df["census_nfd"], df["census_pc"] = False, None, None
    df["paraphrase"] = df["prompt_id"].map(A.paraphrase_of)
    df["condition"] = df["prompt_id"].map(A.condition_of)
    return df


def test_paraphrase_means_known_values_never_fill_and_refuse_mixed_cells():
    it = t1_items(4, variants=("V1",))
    plan = {0: (1, 1, 1), 1: (1, 0, 0), 2: (0, 1, 1), 3: (0, 0, 0)}
    rows = [row(it[i], "nfc", pid, c) for i, cs in plan.items() for pid, c in zip((P0, P1, P2), cs)]
    rows[-1]["error_class"] = "unparseable"
    m = A.paraphrase_means(_frame(rows))
    assert m["correct_mean"].tolist() == [1.0, 1 / 3, 2 / 3, 0.0] and (m["n_paraphrases"] == 3).all()
    assert m["paraphrases"].iloc[0] == ("p0", "p1", "p2") and m["correct_p0"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert m["n_unparseable"].tolist() == [0, 0, 0, 1] and (m["n_rows"] == 3).all() and (m["n_paraphrases_cell"] == "3").all()
    assert set(A.MEANS_KEYS) <= set(m.columns) and "base_pair_id" in m and "source" in m
    # a p0-only item in the same cell: its mean is its single value (never filled in) and the cell is mixed
    extra = row(t1_items(5, variants=("V1",))[4], "nfc", P0, True)
    with pytest.raises(ValueError, match="mix"):
        A.paraphrase_means(_frame([*rows, extra]))
    m2 = A.paraphrase_means(_frame([*rows, extra]), allow_mixed=True)
    assert m2["n_paraphrases"].tolist() == [3, 3, 3, 3, 1] and (m2["n_paraphrases_cell"] == "1|3").all()
    assert not np.isnan(m2.loc[4, "correct_p0"]) and m2.loc[4, "correct_mean"] == 1.0 and m2.loc[4, "paraphrases"] == ("p0",)
    # two rows of one paraphrase for one item are a defect, not something to average
    with pytest.raises(ValueError, match="repeat"):
        A.paraphrase_means(_frame([*rows, rows[0]]))
    # the condition filter keeps one condition
    comp = [row(it[0], "nfc", COMP0, True)]
    assert len(A.paraphrase_means(_frame([*rows, *comp]), condition=A.COMPONENTS_CONDITION)) == 1


def test_cell_table_reports_every_prereg_8_2_field_and_forwards_the_small_cell_rule():
    sim = Sim(3)
    items = t1_items(160, variants=("V1", "V2"))          # 160 base pairs x 2 variants = 320 items, 160 per variant cell
    rows = _e1_rows(items, sim, 0.6)
    for r in [r for r in rows if r["variant"] == "V1"][:30]:
        r["error_class"] = "unparseable"
    df = _frame(rows)
    m = A.paraphrase_means(df)
    tab = A.cell_table(m, n_boot=N_BOOT)
    assert len(tab) == 2 and set(tab["variant"]) == {"V1", "V2"}
    v1 = tab[tab["variant"] == "V1"].iloc[0]
    exp = m.loc[m["variant"] == "V1", "correct_mean"].mean()
    assert v1["estimate"] == pytest.approx(exp) and v1["lo"] < exp < v1["hi"] and v1["ci_method"] in ("bca", "percentile")
    assert not v1["small_cell"] and v1["n_items"] == 160 and v1["n_base_pairs"] == 160 and v1["n_pairs"] is None
    assert v1["n_rows"] == 480 and v1["n_paraphrases"] == 3 and v1["n_paraphrases_set"] == "3" and v1["stratified"]
    assert v1["deff"] >= 1.0 and 0.0 <= v1["icc"] <= 1.0 and v1["unparseable_rate"] == pytest.approx(30 / 480)
    assert not v1["claim_eligible"]                   # 160 items < MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM
    for c in ("estimate", "lo", "hi", "n_items", "n_base_pairs", "deff", "unparseable_rate"):
        assert c in tab.columns
    # a whole-table cell of 320 items is claim-eligible; one cell per model x task when asked
    whole = A.cell_table(m, by=("model", "task"), n_boot=N_BOOT)
    assert whole["n_items"].iloc[0] == 320 and whole["claim_eligible"].iloc[0]
    # a small cell of paraphrase means keeps the percentile interval, flagged; the same cell on p0 rows gets Wilson
    small = m[m["base_pair_id"].isin(sorted(set(m["base_pair_id"]))[:8])]
    s = A.cell_table(small, by=("model", "task"), n_boot=N_BOOT).iloc[0]
    assert s["small_cell"] and s["ci_method"] == "percentile_small_cell" and s["n_base_pairs"] == 8
    p0 = df[(df["paraphrase"] == "p0") & df["base_pair_id"].isin(set(small["base_pair_id"]))]
    w = A.cell_table(p0, by=("model", "task"), n_boot=N_BOOT).iloc[0]
    assert w["small_cell"] and w["ci_method"] == "wilson_deff" and w["value"] == "correct" and w["n_paraphrases"] == 1
    # T3 cells count yes/no pairs; undefined values are dropped and counted
    t3 = _frame([row(it, "nfc", P0, sim.y(it["base_pair_id"], 0.5), pred="yes") for it in t3_items(30)])
    t3["correct"] = t3["correct"].astype(object)
    t3.loc[t3.index[:4], "correct"] = None
    c3 = A.cell_table(t3, by=("model", "task", "variant"), n_boot=N_BOOT)
    # the four undefined rows are the first yes/no pair of each variant: one pair and two items fewer per cell
    assert (c3["n_pairs"] == 29).all() and c3["n_undefined"].sum() == 4 and (c3["n_items"] == 58).all()


# ------------------------------------------------------------------ the primary metric and the all-three rule
def test_primary_metric_weighs_the_four_variant_cells_equally_and_reports_the_range():
    sim = Sim(4)
    probs = {"V1": 0.9, "V2": 0.3, "V3": 0.5, "V4": 0.7}
    rows = []
    for v, p in probs.items():
        n = {"V1": 60, "V2": 20, "V3": 40, "V4": 40}[v]        # unequal cell sizes: the pooled mean is not the equal-weight mean
        rows += _e1_rows(t1_items(n, variants=(v,)), sim, p)
    df = _frame(rows)
    pm = A.primary_metric(df, n_boot=N_BOOT)
    r = pm.iloc[0]
    m = A.paraphrase_means(df)
    equal = m.groupby("variant")["correct_mean"].mean().mean()
    pooled = m["correct_mean"].mean()
    assert r["acc_mean3"] == pytest.approx(equal) and abs(equal - pooled) > 0.02 and r["lo"] < equal < r["hi"]
    assert r["metric"] == A.OPEN_METRIC and not r["is_api"] and r["line"] == "E1_main" and r["n_variants"] == 4
    per_p = [df[df["paraphrase"] == p].groupby("variant")["correct"].mean().mean() for p in ("p0", "p1", "p2")]
    assert [r["acc_p0"], r["acc_p1"], r["acc_p2"]] == pytest.approx(per_p)
    assert r["range_lo"] == pytest.approx(min(per_p)) and r["range_hi"] == pytest.approx(max(per_p)) and r["paraphrases"] == "p0|p1|p2"
    assert r["n_items"] == 160 and r["n_base_pairs"] == 60 and r["n_paraphrases_set"] == "3" and r["min_cell_items"] == 20
    assert r["ci_method"] in ("bca", "percentile") and not r["claim_eligible"]
    # the API path: p0 on the core, the range on the paraphrase subset with p0 restricted to the same items
    core = t1_items(40, variants=("V1", "V2"))
    api_rows = [row(it, "nfc", P0, sim.y(it["base_pair_id"], 0.6)) for it in core]
    sub = core[:30]
    para_rows = [row(it, "nfc", pid, sim.y(it["base_pair_id"], 0.6)) for it in sub for pid in (P1, P2)]
    adf = pd.concat([_frame(api_rows, "api-x", "E1_api_core"), _frame(para_rows, "api-x", "E1_api_paraphrase")], ignore_index=True)
    adf["is_api"] = True
    ap = A.primary_metric(adf, n_boot=N_BOOT).iloc[0]
    assert ap["is_api"] and ap["metric"] == A.API_METRIC and np.isnan(ap["acc_mean3"]) and ap["n_paraphrases_set"] == "1"
    assert ap["acc_p0"] == pytest.approx(pd.DataFrame(api_rows).groupby("variant")["correct"].mean().mean())
    ids = {it["item_id"] for it in sub}
    p0_sub = pd.DataFrame([x for x in api_rows if x["item_id"] in ids]).groupby("variant")["correct"].mean().mean()
    assert ap["acc_p0_subset"] == pytest.approx(p0_sub) and ap["range_on"] == "30 paraphrase-subset items"
    assert ap["range_lo"] <= min(ap["acc_p0_subset"], ap["acc_p1"], ap["acc_p2"]) and ap["n_items"] == 80
    # an API model without the subset reports p0 only
    ap2 = A.primary_metric(_frame(api_rows, "api-y", "E1_api_core").assign(is_api=True), n_boot=N_BOOT).iloc[0]
    assert ap2["range_on"] == "n/a (p0 only)" and ap2["paraphrases"] == "p0"
    # a model without default-condition main rows is reported, never invented
    assert "no E1_main rows" in A.primary_metric(_frame(rows, line="E1_abl_english"), n_boot=N_BOOT).iloc[0]["note"]


def test_direction_holds_under_all_three_paraphrases_and_paraphrase_contrast():
    assert A.direction_holds({"p0": -0.1, "p1": -0.05, "p2": -0.2})["holds"]
    assert A.direction_holds({"p0": -0.1, "p1": -0.05, "p2": -0.2}, direction=-1)["holds"]
    assert not A.direction_holds({"p0": -0.1, "p1": -0.05, "p2": -0.2}, direction=+1)["holds"]
    flip = A.direction_holds({"p0": -0.1, "p1": 0.05, "p2": -0.2})
    assert not flip["holds"] and flip["sign"] is None and "differs" in flip["reason"]
    two = A.direction_holds({"p0": -0.1, "p1": -0.05})
    assert not two["holds"] and two["missing"] == ["p2"] and two["n_paraphrases"] == 2 and two["sign"] == -1
    assert not A.direction_holds({"p0": 0.0, "p1": 0.1, "p2": 0.1})["holds"]
    sim = Sim(5)
    items = t1_items(120, variants=("V1",))
    main = _frame(_e1_rows(items, sim, 0.6))
    abl_rows = []
    for it in items:
        for pid, p in ((P0, 0.4), (P1, 0.4), (P2, 0.4)):
            abl_rows.append(row(it, "nfc", pid, sim.y(it["base_pair_id"], p)))
    abl = _frame(abl_rows, line="E1_abl_zero_shot")
    both = pd.concat([main, abl], ignore_index=True)
    flipped = both.copy()
    mask = (flipped["line"] == "E1_abl_zero_shot") & (flipped["paraphrase"] == "p2")
    flipped.loc[mask, "correct"] = True
    flipped["model"] = "open-f"
    out = A.paraphrase_contrast(pd.concat([both, flipped], ignore_index=True), {"line": "E1_main"}, {"line": "E1_abl_zero_shot"},
                                direction=-1)
    a, f = out.set_index("model").loc["open-a"], out.set_index("model").loc["open-f"]
    assert a["holds"] and a["sign"] == -1 and all(a[f"diff_{p}"] < 0 for p in ("p0", "p1", "p2")) and a["n_p0"] == 120
    assert not f["holds"] and f["diff_p2"] > 0 and f["diff_p0"] < 0


# ------------------------------------------------------------------ tau_m(a) (PREREG 8.5) and the census rows
def _e3_model(root: Path, model: str, sim: Sim, tau: dict, census=None, n_pairs=150, xcopa=True, t3=True):
    """E3_noilai (T1 V1-V4 and T3 on the main file, arms nfc + the keys of `tau`), E3_c2_enriched (T1, nfc +
    placement_new) and E3_xcopa (log-probability outcome) for one model; tau maps arm -> planted effect."""
    base_p = 0.6
    items = t1_items(n_pairs) + (t3_items(60) if t3 else [])
    rows = []
    for it in items:
        rows.append(row(it, "nfc", P0, sim.y(it["base_pair_id"], base_p), pred="yes"))
        for arm, t in tau.items():
            if arm == "placement_new":
                continue
            rows.append(row(it, arm, P0, sim.y(it["base_pair_id"], base_p + t), pred="yes"))
    write_run(root, f"E3_noilai__{model}", model, rows, census=census)
    if "placement_new" in tau:
        c2 = [dict(it, c2_affected=True, in_core=False) for it in t1_items(130, task="T1")]
        for it in c2:
            it["item_id"] = it["item_id"].replace("T1-", "T1c2-")
            it["base_pair_id"] = "c2" + it["base_pair_id"]
        c2_rows = [row(it, "nfc", P0, sim.y(it["base_pair_id"], base_p)) for it in c2]
        c2_rows += [row(it, "placement_new", P0, sim.y(it["base_pair_id"], base_p + tau["placement_new"])) for it in c2]
        write_run(root, f"E3_c2_enriched__{model}", model, c2_rows, item_name="noilai_c2.jsonl", census=census)
    if xcopa:
        xi = xcopa_items(200)
        xrows = []
        for it in xi:
            xrows.append(row(it, "nfc", "xcopa-p0", False, logprob_correct=sim.y(it["item_id"], 0.7), logprob_correct_norm=True))
            for arm, t in tau.items():
                if arm != "placement_new":
                    xrows.append(row(it, arm, "xcopa-p0", False, logprob_correct=sim.y(it["item_id"], 0.7 + t), logprob_correct_norm=True))
        for r in xrows:
            r.pop("base_pair_id", None), r.pop("source", None)
        write_run(root, f"E3_xcopa__{model}", model, xrows, item_name="xcopa_vi_test.jsonl", census=census)


def test_intervention_effects_recover_the_planted_ate_with_a_cluster_bootstrap_ci_and_the_census_rows(tmp_path):
    sim = Sim(6)
    tau = {"nfd": -0.15, "strip_tones": -0.30, "placement_new": -0.10}
    _e3_model(tmp_path, "open-pass", sim, tau, census=censused())
    _e3_model(tmp_path, "open-norm", sim, {"strip_tones": -0.30}, census=censused(nfd="normalizes", pc="normalizes"), xcopa=False)
    _e3_model(tmp_path, "open-corrupt", sim, {"strip_tones": -0.30}, census=censused(nfd="corrupts"), xcopa=False, t3=False)
    L = A.load_runs(tmp_path)
    assert not L.skipped and L.census["open-norm"]["verdict_nfd"] == "normalizes"
    eff = A.intervention_effects(L.rows, census=L.census, n_boot=N_BOOT)
    assert list(eff.columns) == list(A.EFFECT_COLUMNS)
    fam = eff[(eff["model"] == "open-pass") & (eff["cell_kind"] == "family")].set_index(["arm", "task"])
    # the realized family: nfd and strip_tones on T1, T3, XCOPA plus placement_new on the C2 file = 7 of 14
    assert set(fam.index) == {("nfd", "T1"), ("nfd", "T3"), ("nfd", "XCOPA"), ("strip_tones", "T1"), ("strip_tones", "T3"),
                              ("strip_tones", "XCOPA"), ("placement_new", "T1")}
    assert (fam["family_size"] == 7).all() and (fam["family_defined"] == 14).all() and (fam["holm_family"].str.contains("Table 3")).all()
    for arm in ("nfd", "strip_tones"):
        r = fam.loc[(arm, "T1")]
        assert r["effect"] < 0 and r["ci_lo"] < tau[arm] < r["ci_hi"] and r["ci_hi"] < 0 and r["significant"]
        assert r["n_items"] == 600 and r["n_base_pairs"] == 150 and r["n"] == 600 and r["claim_eligible"]
        assert r["p_source"] == "bootstrap" and 0 <= r["p_primary"] <= 1 and 0 <= r["p_mcnemar"] <= 1 and r["p_holm"] >= r["p_primary"]
        assert r["deff"] >= 1.0 and np.isfinite(r["se_paired_t"]) and r["outcome"] == "correct" and r["item_file"] == "main"
    assert fam.loc[("strip_tones", "T1"), "effect"] < fam.loc[("nfd", "T1"), "effect"]
    x = fam.loc[("nfd", "XCOPA")]
    assert x["outcome"] == "logprob_correct" and x["n_base_pairs"] == 200 == x["n_items"] and x["ci_lo"] < tau["nfd"] < x["ci_hi"]
    assert x["p_source"] == "paired_t"                     # 200 items = 200 clusters: the paired t is primary
    c2 = fam.loc[("placement_new", "T1")]
    assert c2["item_file"] == "c2" and c2["c2_affected_only"] and c2["n_items"] == 520 and c2["ci_lo"] < tau["placement_new"] < c2["ci_hi"]
    assert c2["line"] == "E3_c2_enriched" and fam.loc[("nfd", "T1"), "line"] == "E3_noilai"
    # per-variant T1 cells (H3's tau(nfd, T1-V3)) carry no Holm column
    v3 = eff[(eff["model"] == "open-pass") & (eff["cell_kind"] == "variant") & (eff["variant"] == "V3") & (eff["arm"] == "nfd")].iloc[0]
    assert v3["n_items"] == 150 and pd.isna(v3["p_holm"]) and pd.isna(v3["significant"]) and pd.isna(v3["holm_family"])
    assert v3["ci_lo"] < tau["nfd"] < v3["ci_hi"] and np.isfinite(v3["effect"])
    assert set(eff.loc[(eff["model"] == "open-pass") & (eff["cell_kind"] == "variant"), "variant"]) == set(VARIANTS)
    # a normalizing tokenizer: nfd / win1258 cells are 0 by construction (effect 0, no interval, no p, not in the family)
    norm_rows = eff[eff["model"] == "open-norm"]
    zero = norm_rows[norm_rows["zero_by_construction"]].set_index(["arm", "task"])
    assert set(zero.index) == {("nfd", "T1"), ("nfd", "T3"), ("win1258", "T1"), ("win1258", "T3")}
    z = zero.loc[("nfd", "T1")]
    assert z["effect"] == 0.0 and pd.isna(z["ci_lo"]) and pd.isna(z["p_primary"]) and pd.isna(z["significant"]) and z["cell_kind"] == A.ZERO_BY_CONSTRUCTION
    assert "not a test" in z["reason"] and z["n_items"] == 600 and z["acc_base"] == z["acc_arm"]
    nf = norm_rows[norm_rows["cell_kind"] == "family"]
    assert set(nf["arm"]) == {"strip_tones"} and (nf["family_size"] == 2).all()        # T1 and T3 on the main file
    # a corrupting path: refused and reported, never a tested cell
    bad = eff[(eff["model"] == "open-corrupt") & eff["refused"]]
    assert set(bad["arm"]) == {"nfd"} and bad["effect"].isna().all() and "refused" in bad["reason"].iloc[0]
    assert set(eff.loc[(eff["model"] == "open-corrupt") & (eff["cell_kind"] == "family"), "arm"]) == {"strip_tones"}
    # the McNemar column is labelled secondary in the dataclass that produced it; no XCOPA row is ever a generated contrast
    assert (eff.loc[eff["task"] == "XCOPA", "outcome"] == "logprob_correct").all()


def test_intervention_effects_api_models_take_their_baseline_from_e1_api_core_and_have_no_xcopa(tmp_path):
    sim = Sim(7)
    core = t1_items(80, variants=("V1", "V2")) + t3_items(30)
    base_rows = [row(it, "nfc", P0, sim.y(it["base_pair_id"], 0.6), pred="yes") for it in core]
    arm_rows = []
    for it in core:
        arm_rows.append(row(it, "nfd", P0, sim.y(it["base_pair_id"], 0.45), pred="yes"))
        arm_rows.append(row(it, "placement_new", P0, sim.y(it["base_pair_id"], 0.5), pred="yes"))
    write_run(tmp_path, "E1_api_core__api-x", "api-x", base_rows, item_name="noilai_core.jsonl", backend="openai_compat")
    write_run(tmp_path, "E3_api_core__api-x", "api-x", arm_rows, item_name="noilai_api_para300.jsonl", backend="openai_compat")
    L = A.load_runs(tmp_path)
    assert L.rows["is_api"].all()
    eff = A.intervention_effects(L.rows, n_boot=N_BOOT)
    fam = eff[eff["cell_kind"] == "family"].set_index(["arm", "task"])
    assert set(fam.index) == {("nfd", "T1"), ("nfd", "T3"), ("placement_new", "T1"), ("placement_new", "T3")}
    assert (fam["family_size"] == 4).all() and fam["is_api"].all() and "XCOPA" not in set(eff["task"])
    nfd = fam.loc[("nfd", "T1")]
    assert nfd["line"] == "E3_api_core" and nfd["n_items"] == 160 and nfd["effect"] < 0 and nfd["ci_hi"] < 0
    # the C2 arm on the core: only C2-affected items are paired (H4 "on C2-affected items")
    pl = fam.loc[("placement_new", "T1")]
    n_aff = sum(1 for it in core if it["task"] == "T1" and it["c2_affected"])
    assert pl["n_items"] == n_aff and pl["c2_affected_only"] and pd.isna(pl["item_file"])
    assert fam.loc[("placement_new", "T3"), "n_items"] == sum(1 for it in core if it["task"] == "T3" and it["c2_affected"])


# ------------------------------------------------------------------ the explicit-input gap (DD 12.17)
def test_explicit_input_gap_recovers_the_planted_gap_on_core_items_with_cis_and_no_holm():
    sim = Sim(8)
    items = t1_items(300, variants=("V1",)) + t3_items(80)
    main = _frame([row(it, "nfc", pid, sim.y(it["base_pair_id"], 0.5)) for it in items for pid in (P0, P1, P2)])
    core = [it for it in items if it["in_core"]]
    comp = _frame([row(it, "nfc", COMP0, sim.y(it["base_pair_id"], 0.7)) for it in core], line="E1_explicit_input")
    gap = A.explicit_input_gap(pd.concat([main, comp], ignore_index=True), n_boot=N_BOOT)
    assert list(gap["scope"]) == ["all", "T1", "T3"]
    g = gap[gap["scope"] == "all"].iloc[0]
    assert g["n_items"] == len(core) == 310 and g["baseline_line"] == "E1_main" and g["label"] == A.EXPLICIT_INPUT_LABEL
    assert 0.1 < g["gap"] < 0.3 and g["lo"] < 0.2 < g["hi"] and g["lo"] > 0 and g["p_primary"] < 0.01
    assert g["p_source"] == "bootstrap" and g["n_base_pairs"] == 190 and g["ci_method"] in ("bca", "percentile") and g["deff"] >= 1
    assert g["acc_components"] - g["acc_raw"] == pytest.approx(g["gap"]) and np.isfinite(g["se_paired_t"]) and g["stratified"]
    assert not {"p_holm", "significant"} & set(gap.columns)        # estimation only (PREREG 8.3)
    t1 = gap[gap["scope"] == "T1"].iloc[0]
    assert t1["n_items"] == 150 and t1["n_base_pairs"] == 150 and "derived" in t1["label"]
    # without a core baseline the row says so
    nothing = A.explicit_input_gap(pd.concat([main.assign(in_core=False), comp], ignore_index=True), n_boot=N_BOOT)
    assert "no raw-condition core baseline" in nothing.iloc[0]["note"]


# ------------------------------------------------------------------ the T3 headline (DD 5.3)
def test_t3_headline_excludes_spelling_twins_with_partners_and_keeps_the_logprob_modes_apart():
    sim = Sim(9)
    items = t3_items(40, variants=("V1",))
    rows = []
    for it in items:
        c = sim.y(it["base_pair_id"], 0.7)
        rows.append(row(it, "nfc", P0, c, pred=(it["gold"] if c else ("no" if it["gold"] == "yes" else "yes")),
                        forced_choice_correct=bool(sim.rng.random() < 0.9), paired_correct=bool(sim.rng.random() < 0.8),
                        paired_correct_norm=None))
    df = _frame(rows)
    h = A.t3_headline(df, n_boot=N_BOOT)
    assert len(h) == 1
    r = h.iloc[0]
    spelling_pairs = sum(1 for it in items if it["twin_type"] == "spelling")          # one `no` twin per spelling pair
    assert r["n_items"] == 80 and r["n_pairs"] == 40 and r["n_spelling_items"] == 2 * spelling_pairs == 20
    assert r["n_items_excl_spelling"] == 60 and r["n_pairs_excl_spelling"] == 30 and r["n_base_pairs"] == 30
    excl_ids = {it["item_id"] for it in items if it["base_pair_id"] not in {i["base_pair_id"] for i in items if i["twin_type"] == "spelling"}}
    assert r["acc_excl_spelling"] == pytest.approx(df[df["item_id"].isin(excl_ids)]["correct"].mean())
    assert r["acc_all"] == pytest.approx(df["correct"].mean()) and r["lo_excl_spelling"] <= r["acc_excl_spelling"] <= r["hi_excl_spelling"]
    assert r["balanced_accuracy"] == pytest.approx(r["acc_excl_spelling"], abs=0.2) and np.isfinite(r["d_prime"])
    assert r["t3_yesno_lp"] == pytest.approx(df["forced_choice_correct"].mean()) and r["t3_yesno_lp_n"] == 80
    assert r["t3_pair_lp"] == pytest.approx(df["paired_correct"].mean()) and np.isnan(r["t3_pair_lp_norm"]) and r["t3_pair_lp_n"] == 80
    assert r["headline_excludes_twin"] == constants.T3_HEADLINE_EXCLUDES_TWIN == "spelling"
    # no column and no public function contrasts a generated number with a log-probability one (DD 5.3 item 63)
    lp = [c for c in h.columns if "lp" in c.split("_")]
    assert lp and all(c.startswith(("t3_yesno_lp", "t3_pair_lp")) for c in lp)
    assert not any(("diff" in c or "minus" in c or "gap" in c) for c in h.columns)
    public = [n for n in dir(A) if not n.startswith("_") and callable(getattr(A, n))]
    assert not any(("logprob" in n.lower() or "lp" in n.lower().split("_")) and ("generated" in n.lower() or "vs" in n.lower().split("_"))
                   for n in public)


# ------------------------------------------------------------------ writing
def test_write_aggregates_refuses_paper_and_item_level_rows(tmp_path):
    tab = pd.DataFrame({"model": ["a"], "estimate": [0.5], "lo": [np.nan], "n_items": [np.int64(3)], "flag": [True],
                        "paraphrases": [("p0", "p1")]})
    with pytest.raises(ValueError, match="paper/"):
        A.write_aggregates(ROOT / "paper" / "tables", {"t": tab})
    with pytest.raises(ValueError, match="item-level"):
        A.write_aggregates(tmp_path / "out", {"items": pd.DataFrame({"item_id": ["x"], "correct": [1]})})
    assert not (tmp_path / "out").exists()
    written = A.write_aggregates(tmp_path / "out", {"t": tab, "n": {"items": 3}}, label="INTERIM", meta={"seed": 0})
    assert {p.name for p in written["t"]} == {"t.csv", "t.json"} and [p.name for p in written["n"]] == ["n.json"]
    body = json.loads((tmp_path / "out" / "t.json").read_text(encoding="utf-8"))
    assert body["label"] == "INTERIM" and body["n_rows"] == 1 and body["rows"][0]["lo"] is None and body["rows"][0]["n_items"] == 3
    assert body["rows"][0]["paraphrases"] == ["p0", "p1"] and body["meta"] == {"seed": 0}
    assert pd.read_csv(tmp_path / "out" / "t.csv")["estimate"].iloc[0] == 0.5


# ------------------------------------------------------------------ the local echo run
@pytest.mark.skipif(not (ECHO_RUNS / "smoke_echo_20__echo" / "scores.jsonl").exists(), reason="no local echo smoke run")
def test_the_echo_smoke_run_loads_as_a_smoke_run_and_its_clusters_come_from_the_rows_or_the_item_file(tmp_path):
    assert A.load_runs(ECHO_RUNS).skipped and "smoke" in A.load_runs(ECHO_RUNS).skipped[0]["reason"]
    L = A.load_runs(ECHO_RUNS, include_smoke=True)
    assert L.n()["items"] == 20 and L.n()["runs"] == 1 and L.runs[0].model == "echo" and L.runs[0].line == "smoke_echo_20"
    df = L.rows
    assert (df["paraphrase"] == "p0").all() and (df["condition"] == A.DEFAULT_CONDITION).all() and df["base_pair_id"].notna().all()
    cells = A.cell_table(df, n_boot=N_BOOT)
    assert cells["unparseable_rate"].iloc[0] == 1.0 and cells["estimate"].iloc[0] == 0.0 and cells["ci_method"].iloc[0] == "wilson_deff"
    assert cells["n_items"].iloc[0] == 20 and cells["n_base_pairs"].iloc[0] == 20 and cells["small_cell"].iloc[0]
    man = json.loads((ECHO_RUNS / "smoke_echo_20__echo" / "manifest.json").read_text(encoding="utf-8"))
    if A.resolve_item_path(man) is None:
        pytest.skip("the echo run's item file (the local smoke build) is not on this machine")
    rows = [dict(r, base_pair_id=None, source=None) for r in A.read_jsonl(ECHO_RUNS / "smoke_echo_20__echo" / "scores.jsonl")]
    assert "took base_pair_id from the item file" in A.attach_clusters(rows, man)
    assert [r["base_pair_id"] for r in rows] == df["base_pair_id"].tolist() and all(r["source"] in ("lexicon", "pseudo") for r in rows)
