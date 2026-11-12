"""Executable acceptance. Receipts are derived from probes, not supplied PASS flags."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from .reevaluation_gateway import http_json
from .reevaluation_runtime import save, load, checksum, InfrastructureError
from .reevaluation_server import command


def hf_reference(cfg, dep, inv, root):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    request = root/'reference_request.json'
    save(request,{'base':inv['base_path'],'adapters':inv['adapters']})
    output = root/'hf_reference.json.gz'
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=dep['gpu_uuids'][0],NICE_EXECUTION='server',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    result = command([dep['reference_python'],str(Path(__file__).resolve().parents[1]/'scripts/reevaluation_hf_reference.py'),
                      str(request.resolve()),str(output.resolve())],timeout=900,env=env)
    (root/'hf.stdout.log').write_text(result.stdout,encoding='utf-8')
    (root/'hf.stderr.log').write_text(result.stderr,encoding='utf-8')
    return output


def numerics(upstream, reference, cfg):
    with gzip.open(reference,'rt',encoding='utf-8') as f:records=json.load(f)
    errors = []
    for row in records:
        result = http_json(upstream+'/v1/completions',{'model':row['model'],
            'prompt':row['prompt_ids'],'max_tokens':1,'temperature':0,'logprobs':20,
            'return_tokens_as_token_ids':True,'add_special_tokens':False},timeout=120)
        probs = result['choices'][0]['logprobs']['top_logprobs'][0]
        diffs = []
        for token, value in probs.items():
            if not token.startswith('token_id:'):
                raise InfrastructureError('vLLM did not return exact token IDs for numerical acceptance')
            index = int(token.split(':')[1])
            diffs.append(abs(row['logprobs'][index]-value))
        if len(diffs)<20 or max(diffs)>cfg['acceptance']['logprob_atol']:
            raise InfrastructureError('HF/vLLM adapter numerical comparison failed: '+row['model'])
        errors.append({'model':row['model'],'max_abs_logprob_error':max(diffs),'tokens':len(diffs)})
    if len(records)!=15 or {r['model'] for r in records}!=set(cfg['models']):
        raise InfrastructureError('Incomplete five-adapter reference')
    return errors


def capacity(upstream,cfg,token_id):
    g=cfg['generation']; count=g['context_length']-g['max_new_tokens']
    result = http_json(upstream+'/v1/completions',{'model':'initial','prompt':[token_id]*count,
        'max_tokens':g['max_new_tokens'],'temperature':0,'ignore_eos':True,
        'add_special_tokens':False},timeout=cfg['agent']['wall_seconds'])
    usage=result['usage']
    if usage['prompt_tokens']!=count or usage['completion_tokens']!=g['max_new_tokens']:
        raise InfrastructureError('Full context/output capacity was not actually exercised')
    return usage


def accept(cfg,dep,inv,services,backends,reference,root,run_id):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with gzip.open(reference,'rt',encoding='utf-8') as f:token_id=json.load(f)[0]['prompt_ids'][-1]
    from .reevaluation_deploy import probe_network
    save(root/'network.json',probe_network(dep['sdk_image'],dep['docker_network'],
        dep['gateway_url'].rstrip('/')+':'+str(dep['gateway_port_start'])))
    # Every derived image must actually contain the frozen SDK/runtime, not just a label.
    images=sorted({v['image'] for v in dep['task_images'].values()})
    def image_probe(image):
        result=command(['docker','run','--rm','--pull=never','--network','none',
            '--entrypoint','/opt/nice-sdk/bin/python',image,'-c',
            'import importlib.metadata as m,subprocess,shutil,pwd; '
            'pwd.getpwuid(1000); assert all(shutil.which(x) for x in ["bash","git","rg"]); '
            'assert m.version("claude-agent-sdk")=="0.2.87"; '
            'assert subprocess.check_output(["/opt/nice-sdk/bin/claude-pinned","--version"],text=True).split()[0]=="2.1.153"; '
            'print("PASS")'],timeout=60)
        return {'image':image,'stdout':result.stdout}
    with ThreadPoolExecutor(max_workers=8) as pool:
        save(root/'task-runtimes.json',list(pool.map(image_probe,images)))
    def per_service(index):
        result={'numerics':numerics(services.upstreams[index],reference,cfg),
                'capacity':capacity(services.upstreams[index],cfg,token_id),'sdk_routes':[]}
        # Actual tool feedback, all aliases, not a /models name-only check.
        from scripts.smoke_reevaluation import smoke_command
        for model in cfg['models']:
            attempt=root/f'service-{index}'/model
            attempt.mkdir(parents=True,exist_ok=False)
            token,budget=services.gateways[index].register(model,attempt/'budget.json')
            endpoint=dep['gateway_url'].rstrip('/')+':'+str(dep['gateway_port_start']+index)
            env=dict(os.environ,ANTHROPIC_API_KEY=token,ANTHROPIC_AUTH_TOKEN=token)
            name='nice-reeval-accept-'+__import__('uuid').uuid4().hex[:12]
            try:
                args=smoke_command(dep['sdk_image'],dep['docker_network'],endpoint,model,
                                   cfg['claude_code_version'],credentials=env)
                args[2:2]=['--name',name]
                args[2:2]=['--label','nice.run_root='+dep['run_root']]
                try:
                    output=command(args,timeout=660,env=env)
                except InfrastructureError as exc:
                    for stream in ['stdout','stderr']:
                        value=getattr(exc.__cause__,stream,'') or ''
                        if isinstance(value,bytes):value=value.decode(errors='replace')
                        (attempt/(stream+'.log')).write_text(value,encoding='utf-8')
                    raise
                (attempt/'stdout.log').write_text(output.stdout,encoding='utf-8')
                (attempt/'stderr.log').write_text(output.stderr,encoding='utf-8')
                # The smoke emits its explicit receipt only after Read/Write/Bash checks.
                receipt=json.loads(output.stdout.strip().splitlines()[-1])
                if receipt.get('status')!='PASS' or receipt.get('model')!=model:
                    raise InfrastructureError('SDK tool smoke lacks a verified receipt')
                if not receipt.get('thinking_blocks_observed') or not any(
                        r['history_thinking_blocks']>0 for r in budget.records):
                    raise InfrastructureError('Thinking/history retention not observed on adapter route: '+model)
                result['sdk_routes'].append(model)
            finally:
                services.gateways[index].sessions.pop(token,None)
                subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=30)
        save(root/f'service-{index}.json',result)
        return result
    with ThreadPoolExecutor(max_workers=len(backends)) as pool:
        list(pool.map(per_service,range(len(backends))))
    backend=backends[0]
    job={'scope':'development','model':'initial','task_id':dep['development_task']['instance_id']}
    # Positive official reference grading proves the grader; never exposed to the model.
    positive=root/'development-reference';positive.mkdir(exist_ok=False)
    if not backend.grade(job,dep['development_task']['patch'],positive):
        raise InfrastructureError('Official development reference patch failed')
    negative=root/'development-empty';negative.mkdir(exist_ok=False)
    if backend.grade(job,'',negative):
        raise InfrastructureError('Empty patch unexpectedly passed official grading')
    real=root/'development-sdk';real.mkdir(exist_ok=False)
    agent=backend.agent(job,real)
    resolved=backend.grade(job,agent['patch'],real)
    save(real/'result.json',{'resolved':resolved,'agent':agent,'acceptance_only':True})
    if not agent.get('thinking_blocks'):
        raise InfrastructureError('No actual SDK thinking block observed')
    # Stress native compaction without contaminating a formal task or changing budgets.
    context=root/'context-sdk';context.mkdir(exist_ok=False)
    context_job=dict(job,probe='context')
    compact=backend.agent(context_job,context)
    if compact.get('compactions',0)<1:
        raise InfrastructureError('Native SDK context compaction was not observed; formal execution blocked')
    counts=[r['input_tokens'] for r in load(context/'budget.json')['records']]
    if len(counts)<2 or max(counts)-min(counts)<1000:
        raise InfrastructureError('Compaction did not measurably reduce actual backend input tokens')
    files={str(p.relative_to(root)):checksum(p) for p in sorted(root.rglob('*')) if p.is_file()}
    receipt={'status':'PASS','run_id':run_id,'artifacts':files,
             'scope':'real server acceptance; no formal benchmark task executed'}
    save(root/'acceptance.json',receipt)
    return receipt
