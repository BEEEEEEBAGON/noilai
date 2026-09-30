"""Constants module and the count-reconciliation script."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------- divergence test (design 8.8, item 29)
DD = ROOT / "docs" / "DESIGN_DECISIONS.md"
PREREG = ROOT / "docs" / "PREREGISTRATION.md"


def _quotes() -> dict[str, str]:
    """How the binding documents render each constant (f-strings: a constant change breaks the match)."""
    from noilai.stats import e2
    return {
        "MAIN_SAMPLE_PER_CELL": f"{C.MAIN_SAMPLE_PER_CELL} per cell",
        "BOOTSTRAP_B": f"B = {C.BOOTSTRAP_B:,}",
        "BCA_MIN_CLUSTERS": f"≥ {C.BCA_MIN_CLUSTERS} base pairs",
        "PAIRED_T_MIN_BASE_PAIRS": f"≥ {C.PAIRED_T_MIN_BASE_PAIRS} base pairs",
        "SMALL_CELL_MAX_BASE_PAIRS": f"< {C.SMALL_CELL_MAX_BASE_PAIRS} base pairs",
        "HOLM_FAMILY_TABLE3_CELLS": f"{C.HOLM_FAMILY_TABLE3_CELLS} cells",
        "PROBE_SELECTIVITY_MIN": f"δ_sel = {C.PROBE_SELECTIVITY_MIN}",
        "PROBE_SEEDS": f"{C.PROBE_SPLIT_SEEDS} split seeds × {C.PROBE_CONTROL_SEEDS} control-label seeds",
        "PROBE_SYLLABLES_PER_TONE": f"{C.PROBE_SYLLABLES_PER_TONE} per tone",
        "PATCHING_MIN_GAP_NATS": f"{C.PATCHING_MIN_GAP_NATS} nat",
        "PATCHING_MIN_CLEAN_PAIRS": f"{C.PATCHING_MIN_CLEAN_PAIRS} clean pairs",
        "PATCHING_READOUT_B_MIN_RETENTION": f"retention < {C.PATCHING_READOUT_B_MIN_RETENTION}",
        "ATTESTED_EXACT_FLOOR_FOR_H6": f"≥ {C.ATTESTED_EXACT_FLOOR_FOR_H6} exact",
        "MIN_VIABLE_PANEL": f"≥ {C.MIN_VIABLE_PANEL_MODELS} open models spanning ≥ {C.MIN_VIABLE_PANEL_FAMILIES} tokenizer families",
        "H1_MISALIGNED_SPLIT_FLOOR": f"{e2.H1_MISALIGNED_SPLIT_FLOOR} misaligned split syllables",
    }


# constants the pre-registration draft does not quote yet, or quotes with a superseded value (PREREG 8.3 "6 tests",
# 8.9 "5 control-label seeds"; the H6 floor, the readout-B retention fallback and the minimum viable panel are absent).
# Strict xfail: the moment the document is corrected these flip to XPASS and the marker must be removed.
PREREG_STALE = {"HOLM_FAMILY_TABLE3_CELLS", "PROBE_SEEDS", "PATCHING_READOUT_B_MIN_RETENTION", "ATTESTED_EXACT_FLOOR_FOR_H6",
                "MIN_VIABLE_PANEL", "H1_MISALIGNED_SPLIT_FLOOR"}


def test_constants_are_quoted_by_the_binding_documents():
    quotes = _quotes()
    assert len(quotes) >= 15                                  # non-vacuity guard
    dd = DD.read_text(encoding="utf-8")
    missing = {k: v for k, v in quotes.items() if v not in dd}
    assert not missing, f"DESIGN_DECISIONS.md does not quote {missing}"
    assert len(C.HOLM_FAMILY_TABLE3) == C.HOLM_FAMILY_TABLE3_CELLS == 14 and C.PROBE_CONTROL_SEEDS == 3
    assert {c[0] for c in C.HOLM_FAMILY_TABLE3} <= set(C.ARMS) - {"base"}
    assert sum(1 for c in C.HOLM_FAMILY_TABLE3 if c[2] == "c2") == 2 and {c[1] for c in C.HOLM_FAMILY_TABLE3} == {"T1", "T3", "XCOPA"}


@pytest.mark.parametrize("name", sorted(_quotes()))
def test_preregistration_quotes_each_constant(name, request):
    if name in PREREG_STALE:
        request.applymarker(pytest.mark.xfail(strict=True, reason="docs/PREREGISTRATION.md still carries the superseded value or "
                                                                  "omits the constant (paper-docs surface); remove from PREREG_STALE once fixed"))
    assert _quotes()[name] in PREREG.read_text(encoding="utf-8")
