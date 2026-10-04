import json
from pathlib import Path


def test_production_notebook_contains_required_public_gates():
    nb=json.loads(Path('notebooks/kaggle-production-demo.ipynb').read_text())
    text='\n'.join(''.join(c.get('source',[])) if isinstance(c.get('source'),list) else c.get('source','') for c in nb['cells'])
    for needle in ['GPU T4 x2','GPU_T4X2=PASS','MODEL_ORIGINAL_SAFETENSORS=PASS','IPython.display','Audio(','IPA','KAGGLE_PRODUCTION_DEMO=PASS','Vietnamese','code-switch']:
        assert needle in text
