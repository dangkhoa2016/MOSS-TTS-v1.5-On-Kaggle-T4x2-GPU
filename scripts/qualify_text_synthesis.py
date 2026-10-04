#!/usr/bin/env python3
import argparse, json, subprocess, sys
from pathlib import Path
CASES=[('en','English','Today we are testing clear speech on two Tesla T4 GPUs.'),('vi','Vietnamese','Xin chào. Đây là bài kiểm tra tổng hợp giọng nói tiếng Việt trên hai GPU Tesla T4.'),('codeswitch','Vietnamese','Xin chào, đây là Moss TTS on Kaggle, và this sentence switches between Vietnamese and English.')]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model-root',default='/kaggle/input/models'); p.add_argument('--output-dir',default='results/text-synthesis'); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); summary=[]
 for cid,lang,text in CASES:
  wav=out/f'{cid}.wav'; rep=out/f'{cid}.json'; subprocess.run([sys.executable,'scripts/generate.py','--model-root',a.model_root,'--text',text,'--language',lang,'--output',str(wav),'--report',str(rep)],check=True); summary.append(json.loads(rep.read_text()))
 (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)); print('TEXT_SYNTHESIS_QUALIFICATION=PASS')
if __name__=='__main__': main()
