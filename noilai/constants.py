"""Pre-registered constants (design 8.8, red-team item 29): one place for every threshold,
sample size and family definition that the pre-registration, the analysis code and the paper
must agree on. `tests/test_constants.py::test_constants_are_quoted_by_the_binding_documents`
checks that docs/DESIGN_DECISIONS.md (binding) quotes these values and marks, per constant,
where docs/PREREGISTRATION.md still carries a superseded value; a change here needs a dated
row in docs/DEVIATIONS.md. `noilai/stats/tables.py` reads HOLM_FAMILY_TABLE3 and
`noilai/stats/bootstrap.py` reads the interval constants below (design 8.8, item 29).
"""
from __future__ import annotations

# --- E1 / sampling --------------------------------------------------------------------------
T1_CELLS = ("V1", "V2", "V3", "V4")
MAIN_SAMPLE_PER_CELL = 350          # open-model main sample: 350 items per task x variant cell (4,200)
CORE_PER_CELL = 125                 # API core: 125 per T1/T2 cell, 62 yes/no pairs per T3 cell
API_PARAPHRASE_SUBSET = 300         # core items under each of p1, p2 for API models
PILOT_ITEMS = 200                   # Gate 1 pilot per model
PILOT_MODELS = 3
PILOT_GO_MIN_GAIN_OVER_COPY = 0.15  # go/no-go: >= 15 points over the copy baseline for >= 1 pilot model, CI excluding 0
PILOT_MAX_DISCORDANCE = 0.35        # above this, cell-level claims are dropped or n raised BEFORE test runs
C2_ENRICHED_MIN_ITEMS = 500
HUMAN_BASELINE_PEOPLE = 20
HUMAN_BASELINE_ITEMS_PER_PERSON = 30
HUMAN_BASELINE_ANCHORS = 6
HUMAN_BASELINE_DOUBLE_CODED = 240
VALIDATION_ITEMS = 1000
VALIDATION_OVERLAP = 200
ATTESTED_EXACT_FLOOR_FOR_H6 = 100   # verified exact two-syllable rows from >= 3 collections, else H6 is descriptive
ATTESTED_MIN_COLLECTIONS = 3

# --- scoring ---------------------------------------------------------------------------------
PRIMARY_METRIC = "strict T1 accuracy, averaged over the three paraphrases, pooled over V1-V4 with equal weight, per model"
T3_HEADLINE_EXCLUDES_TWIN = "spelling"
MAX_NEW_TOKENS = 64
THINKING_CHARS_ALLOWED_IN_MAIN_RUNS = 0

# --- statistics ------------------------------------------------------------------------------
ALPHA = 0.05
BOOTSTRAP_B = 2000
BOOTSTRAP_CLUSTER = "base_pair_id"
BCA_MIN_CLUSTERS = 50               # BCa interval when a cell has >= 50 base pairs, else percentile (8.2)
PAIRED_T_MIN_BASE_PAIRS = 200       # paired t on base-pair-level mean differences is primary from 200 base pairs (8.2)
SMALL_CELL_MAX_BASE_PAIRS = 20      # cells with < 20 base pairs (or at 0 % / 100 %): Wilson on n / DEFF (8.2)
# Table 3 Holm family, one per model row, enumerated exactly (design 8.3, item 65): (arm, task, item file);
# arm names are the docs/DATA_FORMAT.md names of ARMS (the runner's aliases: base -> nfc, pc -> win1258).
HOLM_FAMILY_TABLE3 = tuple((arm, task, "main") for arm in ("nfd", "pc", "strip_tones", "strip_all")
                           for task in ("T1", "T3", "XCOPA")) + (("placement_new", "T1", "c2"), ("placement_new", "T3", "c2"))
HOLM_FAMILY_TABLE3_CELLS = 14
assert len(HOLM_FAMILY_TABLE3) == HOLM_FAMILY_TABLE3_CELLS
MIN_ITEMS_PER_CELL_FOR_10_POINT_CLAIM = 300
TOST_EQUIVALENCE_MARGIN_POINTS = 2.0

# --- E4 --------------------------------------------------------------------------------------
PROBE_SELECTIVITY_MIN = 0.15        # delta_sel, provisional; frozen from the pilot before the main extraction
PROBE_TEST_FRAC = 0.30
PROBE_SPLIT_SEEDS = 5
PROBE_CONTROL_SEEDS = 3             # design 9.2 primary grid: 5 split seeds x 3 control-label seeds (2,800 fits on the 4B)
PROBE_SPLIT_FRACS = (0.6, 0.1, 0.3)   # nested split by syllable identity: train / dev (tunes L2) / test (9.2)
PROBE_L2_GRID = (0.01, 0.1, 1.0, 10.0)   # inverse regularization C tuned once on the dev fold at one mid layer (9.2)
PROBE_SYLLABLES_PER_TONE = 300
PROBE_MIN_CARRIERS = 4
PATCHING_MIN_GAP_NATS = 1.0         # filter 3: LD_clean - LD_corrupt >= 1 nat and greedy clean answer correct
PATCHING_MIN_CLEAN_PAIRS = 200
PATCHING_READOUT_B_MIN_RETENTION = 50   # below this, readout A only (localization study)
STEERING_ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)

# --- re-encoding arms ------------------------------------------------------------------------
BASELINE_PLACEMENT = "old"          # stored form and C2 baseline (hòa); flips only via DEVIATIONS.md after the corpus count
ARMS = ("base", "nfd", "pc", "placement_new", "strip_tones", "strip_all")
ARM_SCOPE_PRIMARY = "whole_prompt"
BF16_DRIFT_CHECK_ITEMS = 200

# --- panel -----------------------------------------------------------------------------------
MIN_VIABLE_PANEL_MODELS = 8
MIN_VIABLE_PANEL_FAMILIES = 4


def as_dict() -> dict:
    return {k: v for k, v in globals().items() if k.isupper()}
