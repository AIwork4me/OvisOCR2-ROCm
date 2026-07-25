PLATFORM ?= linux-rocm
VERSION  ?= v16
REVISION ?= 2b161d0
MODEL_ID ?= ovisocr2
# Optional backend/server config forwarded to the adapter (empty = use adapter_config.py defaults).
BACKEND       ?=
SERVER_URL    ?=
API_MODEL_NAME ?=
# Clean full run by default: CDM scoring ON, --skip-existing OFF.
# Override: `make eval-linux RESUME=1` to resume an interrupted run; `CDM=0` to
# disable CDM scoring for a quick debug score.
CDM ?= 1
RESUME ?= 0
CDM_FLAG = $(if $(filter 1,$(CDM)),--cdm,)
RESUME_FLAG = $(if $(filter 1,$(RESUME)),--skip-existing,)

setup-linux:
	bash adapter/setup/00-install-deps.sh
setup-windows:
	powershell -ExecutionPolicy Bypass -File adapter\setup\00-install-deps.ps1

demo:
	OUT=$$(mktemp -d); omnidocbench-rocm infer --adapter adapter/run_adapter.py --img-dir examples --out-dir $$OUT --platform $(PLATFORM); ls $$OUT

eval-linux:
	omnidocbench-rocm run --stage all --platform linux-rocm --version $(VERSION) --revision $(REVISION) \
	  --adapter adapter/run_adapter.py --model-id $(MODEL_ID) \
	  $(if $(BACKEND),--backend $(BACKEND)) \
	  $(if $(SERVER_URL),--server-url $(SERVER_URL)) \
	  $(if $(API_MODEL_NAME),--api-model-name $(API_MODEL_NAME)) \
	  --git-commit $$(git rev-parse HEAD) --results-dir results/omnidocbench/$(VERSION)/linux-rocm \
	  $(CDM_FLAG) $(RESUME_FLAG)

eval-windows:
	omnidocbench-rocm run --stage all --platform windows-hip --version $(VERSION) --revision $(REVISION) \
	  --adapter adapter/run_adapter.py --model-id $(MODEL_ID) \
	  $(if $(BACKEND),--backend $(BACKEND)) \
	  $(if $(SERVER_URL),--server-url $(SERVER_URL)) \
	  $(if $(API_MODEL_NAME),--api-model-name $(API_MODEL_NAME)) \
	  --git-commit $$(git rev-parse HEAD) --results-dir results/omnidocbench/$(VERSION)/windows-hip \
	  $(CDM_FLAG) $(RESUME_FLAG)

publish:
	omnidocbench-rocm conformance . && echo CONFORMANT

smoke-test:
	python -m pytest
