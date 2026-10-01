PY ?= /home/user/venv-noilai/bin/python
# RELEASE is the run plan's `release:` key (DESIGN_DECISIONS 12.30: v0.1 superseded; v0.3 at the generator freeze). No trailing
# comment on the assignment: make keeps the blanks before a comment in the value, and `$(RELEASE)/file` then splits into two words.
RELEASE ?= data/release/v0.3
# SEED has NO default: the build seed is private (DESIGN_DECISIONS 4.6, item 51) and lives in $(RELEASE)/manifest_private.json;
# run `make data SEED=<from the private release manifest>`. The sampling seeds below are public: they select from the built test split.

.PHONY: test resources data attested audit validation baseline lint

test:
	$(PY) -m pytest -q

resources:
	$(PY) scripts/fetch_resources.py

data: resources
	@test -n "$(SEED)" || { echo "make data: SEED is unset. The build seed is private (DESIGN_DECISIONS 4.6); run: make data SEED=<from the private release manifest>" >&2; exit 1; }
	$(PY) scripts/build_data.py --out $(RELEASE) --seed $(SEED)
	$(PY) scripts/build_attested.py --out $(RELEASE)/attested.jsonl
	$(PY) scripts/sample_items.py main --release $(RELEASE) --per-cell 350 --seed 20261201
	$(PY) scripts/sample_items.py c2 --release $(RELEASE) --n 500 --seed 20261202

audit: resources
	$(PY) scripts/audit_tokenizers.py --spm data/external/gemma3_tokenizer.model:gemma3 data/external/gemma2_tokenizer.model:gemma2 --out data/audit

validation:
	$(PY) scripts/make_validation_forms.py sample --items $(RELEASE)/noilai_test.jsonl --dev $(RELEASE)/noilai_dev.jsonl --out data/validation --n 1000 --overlap 200 --validators A B C

# Human-baseline forms, DESIGN_DECISIONS 10.2: 20 forms x 30 items from the open-model main sample (6 anchors on every form,
# 240 items each on exactly two forms = 246 distinct items). The coverage printout of the builder is checked against that
# design; forms that do not meet it are deleted and the target fails (item 66; the design is docs/HUMAN_BASELINE_FORM.md section 1).
# 20261102 is the public sampling seed of that snippet, never the build seed.
baseline:
	$(PY) scripts/make_validation_forms.py baseline --items $(RELEASE)/noilai_main.jsonl --out data/human --n-forms 20 --per-form 30 --seed 20261102 \
	  | $(PY) -c "import json, sys; d = json.loads(sys.stdin.read().strip().splitlines()[-1]); print(d); ok = d['forms'] == 20 and d['per_form'] == 30 and d['distinct_items'] == 246 and d['min_appearances'] == 2 and d['max_appearances'] == 20; sys.exit(0 if ok else 1)" \
	  || { echo "make baseline: the forms are NOT the DESIGN_DECISIONS 10.2 design (20 x 30, 6 anchors seen by all 20, 240 items on exactly two forms); deleted, do not send. The design is docs/HUMAN_BASELINE_FORM.md section 1." >&2; rm -f data/human/baseline_form_*.csv; exit 1; }

lint:
	$(PY) -m ruff check noilai scripts tests
