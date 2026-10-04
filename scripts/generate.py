#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import torch, torchaudio
from transformers import AutoModel, AutoProcessor
from moss_t4x2.device_map import build_device_map
from moss_t4x2.generation import generation_plan, processor_load_kwargs
from moss_t4x2.metrics import generation_metrics, validate_waveform
from moss_t4x2.model_discovery import discover_model


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-root', default='/kaggle/input/models')
    ap.add_argument('--text', required=True); ap.add_argument('--language', default='English')
    ap.add_argument('--output', required=True); ap.add_argument('--report', required=True)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--max-new-tokens', type=int, default=256)
    a=ap.parse_args(); plan=generation_plan(a.text,a.language,seed=a.seed,max_new_tokens=a.max_new_tokens)
    model_dir=discover_model(Path(a.model_root)); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    processor=AutoProcessor.from_pretrained(str(model_dir),**processor_load_kwargs())
    conv=[[processor.build_user_message(text=a.text,language=a.language)]]; batch=processor(conv,mode='generation')
    input_ids=batch['input_ids'].cpu(); attention_mask=batch['attention_mask'].cpu()
    if getattr(processor,'audio_tokenizer',None) is not None: processor.audio_tokenizer=processor.audio_tokenizer.cpu()
    del processor; gc.collect(); torch.cuda.empty_cache()
    for i in range(2):
        with torch.cuda.device(i): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter(); model=AutoModel.from_pretrained(str(model_dir),trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True,device_map=build_device_map())
    load_s=time.perf_counter()-t; model.eval(); input_ids=input_ids.to('cuda:0'); attention_mask=attention_mask.to('cuda:0')
    torch.cuda.synchronize(); t=time.perf_counter()
    with torch.inference_mode():
        outputs=model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=a.max_new_tokens,audio_temperature=1.7,audio_top_p=0.8,audio_top_k=25,audio_repetition_penalty=1.0)
    torch.cuda.synchronize(); gen_s=time.perf_counter()-t
    cpu_outputs=[(int(start),ids.detach().cpu()) for start,ids in outputs]
    peaks=[]
    for i in range(2):
        with torch.cuda.device(i): peaks.append(torch.cuda.max_memory_allocated()/1024**3)
    del outputs,model,input_ids,attention_mask; gc.collect(); torch.cuda.empty_cache()
    t=time.perf_counter(); processor=AutoProcessor.from_pretrained(str(model_dir),**processor_load_kwargs()); processor.audio_tokenizer=processor.audio_tokenizer.to('cuda:0')
    with torch.inference_mode(): messages=processor.decode(cpu_outputs)
    decode_s=time.perf_counter()-t; audio=messages[0].audio_codes_list[0].detach().float().cpu(); duration=audio.numel()/24000.0
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); torchaudio.save(str(out),audio.unsqueeze(0),24000)
    signal=validate_waveform(audio.tolist()); report={**plan,"model_dir":str(model_dir),"model_load_seconds":load_s,"sample_rate":24000,"num_samples":audio.numel(),"wav_path":str(out),"signal":signal,**generation_metrics(generate_seconds=gen_s,decode_seconds=decode_s,audio_duration_seconds=duration,gpu_peaks_gib=peaks)}
    rp=Path(a.report); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2,ensure_ascii=False)); print(json.dumps(report,indent=2,ensure_ascii=False)); print('MOSS_TTS_GENERATION=PASS')

if __name__=='__main__': main()
