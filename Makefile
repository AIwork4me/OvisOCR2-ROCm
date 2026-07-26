PLATFORM ?= linux-rocm
VERSION  ?= v16
REVISION ?= 2b161d0
MODEL_ID ?= ovisocr2
VENV     ?= /root/venvs/vllm-0221b
# Real eval defaults to the vLLM backend; smoke is opt-in via demo-smoke.
BACKEND  ?= vllm
CDM ?= 1
RESUME ?= 0
CDM_FLAG = $(if $(filter 1,$(CDM)),--cdm,)
RESUME_FLAG = $(if $(filter 1,$(RESUME)),--skip-existing,)

PY = $(VENV)/bin/python

.PHONY: install-dev setup-linux check smoke-test demo-smoke demo-real eval-linux eval-windows conformance build clean publish

install-dev:
	pip install -e ".[dev]"

setup-linux:
	VENV=$(VENV) bash adapter/setup/00-install-deps.sh

check:
	ruff check . && pytest -q && python -m build && omnidocbench-rocm conformance .

smoke-test:
	python -m pytest

demo-smoke:
	omnidocbench-rocm infer --adapter adapter/run_adapter.py --img-dir examples --out-dir $$(mktemp -d) --platform $(PLATFORM) --backend smoke

demo-real:
	HIP_VISIBLE_DEVICES=0 $(PY) adapter/run_adapter.py --img-dir examples --out-dir /tmp/ovisocr2-demo --platform linux-rocm --backend vllm --limit-pages 1

eval-linux:
	omnidocbench-rocm run --stage all --platform linux-rocm --version $(VERSION) --revision $(REVISION) \
	  --adapter adapter/run_adapter.py --model-id $(MODEL_ID) --backend $(BACKEND) \
	  --git-commit $$(git rev-parse HEAD) --results-dir results/omnidocbench/$(VERSION)/linux-rocm \
	  $(CDM_FLAG) $(RESUME_FLAG)

eval-windows:
	@echo "windows-hip real inference is unsupported (community-wanted: no Qwen3-Next GDN HIP-SDK path)."; exit 1

conformance:
	omnidocbench-rocm conformance . && echo CONFORMANT

build:
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache adapter/__pycache__ tests/__pycache__
	find . -name '*.pyc' -delete

publish: conformance
