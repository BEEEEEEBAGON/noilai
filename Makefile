PY ?= /home/user/venv-noilai/bin/python
RELEASE ?= data/release/v0.1
SEED ?= 20261004

.PHONY: test resources data attested audit validation baseline lint

test:
	$(PY) -m pytest -q

resources:
	$(PY) scripts/fetch_resources.py

data: resources
	$(PY) scripts/build_data.py --out $(RELEASE) --seed $(SEED)
	$(PY) scripts/build_attested.py --out $(RELEASE)/attested.jsonl

audit: resources
	$(PY) scripts/audit_tokenizers.py --spm data/external/gemma3_tokenizer.model:gemma3 data/external/gemma2_tokenizer.model:gemma2 --out data/audit

validation:
	$(PY) scripts/make_validation_forms.py sample --items $(RELEASE)/noilai_test.jsonl --dev $(RELEASE)/noilai_dev.jsonl --out data/validation --n 1000 --overlap 200 --validators A B C

baseline:
	$(PY) scripts/make_validation_forms.py baseline --items $(RELEASE)/noilai_core.jsonl --out data/human --n-forms 20 --per-form 40

lint:
	$(PY) -m ruff check noilai scripts tests
