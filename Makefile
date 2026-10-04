.PHONY: test audit

test:
	PYTHONPATH=. python -m pytest -q

audit:
	PYTHONPATH=. python -m pytest -q
	python -m compileall -q moss_t4x2 scripts
	python -m json.tool notebooks/kaggle-production-demo.ipynb >/dev/null
	git diff --check
	@if git ls-files | grep -Ei '\.(safetensors|gguf|pth|pt|onnx)$$'; then echo 'tracked model weight artifact found'; exit 1; else echo 'MODEL_WEIGHTS_TRACKED=NO'; fi
