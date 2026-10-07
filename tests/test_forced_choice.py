"""EXPLORATORY T1 forced choice (noilai.eval.forced_choice; docs/FORCED_CHOICE_EXPLORATORY.md)."""
import json
import subprocess
import sys
from pathlib import Path

from noilai.eval import forced_choice as FC
from noilai.gen import variants as V
from noilai.gen.generate import syl_dict
from noilai.validation import phrase_syllables, spell_pair
from noilai.vi.reencode import canonical_text

ROOT = Path(__file__).resolve().parents[1]


def _t1(phrase: str, variant: str) -> dict:
    a, b = phrase_syllables(phrase)
    g = V.apply(variant, a, b)
    return {"task": "T1", "variant": variant, "input": phrase, "input_syllables": [syl_dict(a), syl_dict(b)],
            "gold": [spell_pair(g)], "gold_syllables": [syl_dict(g[0]), syl_dict(g[1])], "item_id": "T1-x"}


def test_candidate_set_is_built_by_rule_and_never_contains_a_correct_answer_twice():
    it = _t1("trung bình", "V1")
    c = FC.t1_candidates(it)
    assert c[0] == {"label": "gold", "text": "trinh bùng"}
    gold = set(canonical_text(c[0]["text"]).split())
    texts = [canonical_text(x["text"]) for x in c]
    assert len(set(texts)) == len(texts)                                    # deduplicated
    assert all(set(t.split()) != gold for x, t in zip(c[1:], texts[1:]) if x["label"] != "spelling")   # no lenient-correct
    labels = [x["label"] for x in c]
    assert "copy" in labels and "reversal" in labels and "wrong_variant:V4" in labels and "wrong_variant:V6" not in labels


def test_spelling_distractor_only_with_a_trigger_and_kept_out_of_the_headline():
    it = _t1("cá mè", "V1")                         # gold ké mà: the k trigger
    c = FC.t1_candidates(it)
    sp = [x for x in c if x["label"] == "spelling"]
    assert sp and sp[0]["text"].startswith("cé")
    fc = {"labels": [x["label"] for x in c], "lp": [-1.0] + [-5.0] * (len(c) - 2) + [-0.5], "n_tokens": [2] * len(c)}
    m = FC.fc_metrics(fc)
    assert m["fc_correct"] is True and m["fc_n"] == len(c) - 1 and m["fc_spelling_win"] is False


def test_metrics_rank_ties_and_missing():
    fc = {"labels": ["gold", "copy", "reversal"], "lp": [-3.0, -2.0, -4.0], "n_tokens": [4, 2, 2]}
    m = FC.fc_metrics(fc)
    assert m["fc_correct"] is False and m["fc_rank"] == 2 and m["fc_chance"] == 1 / 3
    assert m["fc_correct_mean"] is True                     # per token: gold -0.75 beats copy -1.0 and reversal -2.0
    tie = FC.fc_metrics({"labels": ["gold", "copy"], "lp": [-2.0, -2.0], "n_tokens": [1, 1]})
    assert tie["fc_correct"] is False                                        # a tie is not a win
    assert FC.fc_metrics({"labels": ["gold", "copy"], "lp": [None, -2.0], "n_tokens": [1, 1]})["fc_correct"] is None
    assert FC.fc_metrics(None)["fc_correct"] is None


def test_check_run_end_to_end_on_the_echo_backend(tmp_path):
    rel = tmp_path / "rel"
    py = sys.executable
    subprocess.run([py, "scripts/build_data.py", "--out", str(rel), "--seed", "5", "--n-lexicon", "60", "--n-pseudo", "30",
                    "--per-cell-t1", "10", "--per-cell-t2", "5", "--per-cell-t3", "5", "--core-per-cell", "2"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run([py, "scripts/run_eval.py", "--items", str(rel / "noilai_dev.jsonl"), "--backend", "echo", "--smoke",
                    "--limit", "12", "--tasks", "T1", "--arms", "nfc", "nfd", "--run-id", "echo_t1", "--out-root", str(tmp_path / "runs"),
                    "--score"], cwd=ROOT, check=True, capture_output=True, text=True)
    run = tmp_path / "runs" / "echo_t1"
    r = subprocess.run([py, "scripts/check_run.py", "--run", str(run), "--n-boot", "200"], cwd=ROOT, check=True, capture_output=True, text=True)
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["ok"] and rep["checks"] == {"item_file_matches_manifest": True, "rescoring_identical": True}
    st = json.loads((run / "stats.json").read_text())
    assert "T1|nfc" in st["by_task_arm"] and "T1|nfd" in st["paired_vs_nfc"]
    v = subprocess.run([py, "scripts/check_run.py", "--run", str(run), "--verify"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert v.returncode == 0
    (run / "scores.jsonl").write_text((run / "scores.jsonl").read_text() + "\n")          # tamper: one byte
    v2 = subprocess.run([py, "scripts/check_run.py", "--run", str(run), "--verify"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert v2.returncode == 1 and "scores.jsonl" in v2.stdout
