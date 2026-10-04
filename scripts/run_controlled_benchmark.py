#!/usr/bin/env python3
import argparse, json, statistics, subprocess, sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model-root',default='/kaggle/input/models'); p.add_argument('--runs',type=int,default=3); p.add_argument('--output-dir',default='results/benchmark'); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=[]
 text='This controlled benchmark measures original BF16 Moss TTS generation on two Tesla T4 GPUs.'
 for i in range(a.runs):
  wav=out/f'run-{i+1}.wav'; rep=out/f'run-{i+1}.json'; subprocess.run([sys.executable,'scripts/generate.py','--model-root',a.model_root,'--text',text,'--language','English','--output',str(wav),'--report',str(rep)],check=True); rows.append(json.loads(rep.read_text()))
 summary={'runs':a.runs,'median_generation_rtf':statistics.median(r['generation_rtf'] for r in rows),'median_generate_seconds':statistics.median(r['generate_seconds'] for r in rows),'rows':rows}; (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); print('CONTROLLED_BENCHMARK=PASS')
if __name__=='__main__': main()
