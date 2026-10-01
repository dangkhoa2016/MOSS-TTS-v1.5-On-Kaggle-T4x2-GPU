.PHONY: test audit

test:
	PYTHONPATH=. python -m pytest -q

audit:
	PYTHONPATH=. python -m pytest -q
	python -m compileall -q moss_t4x2 scripts
	python -m json.tool notebooks/kaggle-production-demo.ipynb >/dev/null
	git diff --check
	PYTHONPATH=. python -m moss_t4x2.audit