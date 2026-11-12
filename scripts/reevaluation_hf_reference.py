"""Short HF/PEFT reference only. Execute on an authorized server GPU, before vLLM."""
import gzip
import json
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    if os.environ.get('NICE_EXECUTION') != 'server' or not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):
        raise SystemExit('Server and explicit authorized GPU UUID required')
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(request['base'],local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(request['base'],local_files_only=True,
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    records = []
    model = PeftModel.from_pretrained(model,request['adapters']['initial']['path'],adapter_name='initial',is_trainable=False)
    for name, record in request['adapters'].items():
        if name != 'initial':model.load_adapter(record['path'],adapter_name=name,is_trainable=False)
    model.eval()
    for name in request['adapters']:
        model.set_adapter(name)
        for text in ['Write a Python function that adds two integers.',
                     'Explain why an empty list is false in Python.',
                     'Fix: def square(x): return x + x']:
            ids = tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,
                add_generation_prompt=True,enable_thinking=True,preserve_thinking=True,reasoning_effort='xhigh')
            inputs = torch.tensor([ids],device='cuda:0')
            with torch.inference_mode():
                logits = model(input_ids=inputs).logits[0,-1].float().log_softmax(-1).cpu()
            records.append({'model':name,'prompt_ids':ids,'logprobs':logits.tolist()})
    with gzip.open(sys.argv[2],'wt',encoding='utf-8') as f:json.dump(records,f)
