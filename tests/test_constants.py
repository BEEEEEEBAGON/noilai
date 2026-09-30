"""Constants module and the count-reconciliation script."""
import json
import subprocess
import sys
from pathlib import Path

from noilai import constants as C

ROOT = Path(__file__).resolve().parents[1]


def test_constants_are_consistent():
    d = C.as_dict()
    assert d["MAIN_SAMPLE_PER_CELL"] == 350 and d["CORE_PER_CELL"] == 125 and d["PILOT_ITEMS"] == 200
    assert 0 < C.PROBE_SELECTIVITY_MIN < 1 and C.BASELINE_PLACEMENT in ("old", "new")
    assert C.T3_HEADLINE_EXCLUDES_TWIN == "spelling" and C.ARM_SCOPE_PRIMARY in ("whole_prompt", "item")
    assert C.MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM <= C.MAIN_SAMPLE_PER_CELL
    assert C.HUMAN_BASELINE_ANCHORS + C.HUMAN_BASELINE_DOUBLE_CODED == 246


def test_reconcile_counts_matches_the_documented_inventory_facts(tmp_path):
    out = tmp_path / "counts.json"
    subprocess.run([sys.executable, "scripts/reconcile_counts.py", "--out", str(out)], cwd=ROOT, check=True, capture_output=True, text=True)
    c = json.loads(out.read_text())
    h, inv = c["hunspell"], c["inventory"]
    assert h["lowercase_letter_entries"] == 6611 and h["parsable"] == 6595 and h["placement_differing_syllables"] == 69
    assert set(h["rejected_phonotactics"]) == {"gip", "têt", "xit"}
    assert inv["base_rimes"] == inv["extended_rimes"] == 162 and inv["extension_added"] > 100
    assert c["distributions_new_file"]["tone"]["sac"] > c["distributions_new_file"]["tone"]["nga"]
    if "gemma3" in c["tokenizer_audit"]:
        g = c["tokenizer_audit"]["gemma3"]["summary"]
        assert 1.7 < g["nfc"]["tokens_per_syllable_mean"] < 1.9 and 2.7 < g["nfd"]["tokens_per_syllable_mean"] < 2.95
        assert c["tokenizer_audit"]["gemma3"]["normalizes_nfd"] is False
