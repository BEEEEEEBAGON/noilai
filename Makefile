PY ?= /home/user/venv-noilai/bin/python
RELEASE ?= data/release/v0.2        # the run plan's `release:` key (DESIGN_DECISIONS 12.30: v0.1 superseded; v0.3 at the generator freeze)
SEED ?= 20261004

.PHONY: test resources data attested audit validation baseline lint

test:
	$(PY) -m pytest -q

resources:
	$(PY) scripts/fetch_resources.py

data: resources
	$(PY) scripts/build_data.py --out $(RELEASE) --seed $(SEED)
	$(PY) scripts/build_attested.py --out $(RELEASE)/attested.jsonl
	$(PY) scripts/sample_items.py main --release $(RELEASE) --per-cell 350 --seed 20261004
	$(PY) scripts/sample_items.py c2 --release $(RELEASE) --n 500 --seed 20261005

audit: resources
	$(PY) scripts/audit_tokenizers.py --spm data/external/gemma3_tokenizer.model:gemma3 data/external/gemma2_tokenizer.model:gemma2 --out data/audit

validation:
	$(PY) scripts/make_validation_forms.py sample --items $(RELEASE)/noilai_test.jsonl --dev $(RELEASE)/noilai_dev.jsonl --out data/validation --n 1000 --overlap 200 --validators A B C

baseline:
	@echo "human-baseline forms: the DESIGN_DECISIONS 10.2 design (20 x 30, 6 anchors, exact double coverage of 240 items from $(RELEASE)/noilai_main.jsonl)"
	@echo "is not yet in scripts/make_validation_forms.py baseline (item 66); build the forms with the snippet in docs/HUMAN_BASELINE_FORM.md."
	@echo "The superseded 20 x 40 core-item design is refused here so that it cannot be sent by mistake."
	@exit 1

lint:
	$(PY) -m ruff check noilai scripts tests
