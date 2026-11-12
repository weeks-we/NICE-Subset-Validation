"""Separate SDK reevaluation entrypoint; existing training entrypoints stay unchanged."""
from __future__ import annotations
import argparse
import ipaddress
import json
import os
from pathlib import Path
import platform
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from nice_validation.reevaluation_runtime import (Journal, MockBackend, jobs_for, schedule,
    report, exclusive, load, save, checksum)
from nice_validation.reevaluation_deploy import independent_paths, inventory, inspect_image


def validate(cfg):
    if (cfg.get('execution')!='server' or cfg.get('agent_interface')!='claude_agent_sdk'
        or cfg.get('vllm_version')!='0.22.0' or cfg.get('claude_code_version')!='2.1.153'
        or cfg.get('claude_agent_sdk_version')!='0.2.87'):
        raise ValueError('Unsupported runtime/version configuration')
    expected={'initial','low_lr1_p100','low_lr2_p025','high_lr1_p050','high_lr2_p100'}
    if len(cfg['models'])!=5 or set(cfg['models'])!=expected:
        raise ValueError('The original five selected adapters must be preserved')
    if cfg['scopes']!={'benchmark':100,'nice_retest':20}:
        raise ValueError('Frozen evaluation scopes must be 100 + 20')
    for section,fields in [('generation',['context_length','max_new_tokens']),
                           ('agent',['max_turns','wall_seconds','total_output_tokens','tool_seconds','grading_seconds']),
                           ('resources',['gpu_count','tensor_parallel_size','slots_per_service'])]:
        for field in fields:
            if type(cfg[section][field]) is not int or cfg[section][field]<=0:
                raise ValueError('Positive integer required: '+field)
    if cfg['agent']['max_turns']!=250 or cfg['resources']['slots_per_service']!=1:
        raise ValueError('Keep 250 turns and one task slot per inference service')
    if cfg['generation']['max_new_tokens']>=cfg['generation']['context_length']:
        raise ValueError('Output reserve must leave room for input')
    if cfg['agent']['model_failure_retries']!=0 or cfg['agent']['infrastructure_retries']!=2:
        raise ValueError('Keep zero model retries and at most two infrastructure retries')
    if not 0<cfg['acceptance']['logprob_atol']<=0.2:
        raise ValueError('Numerical tolerance must be frozen before any formal task')


def source_identity():
    root=Path(__file__).resolve().parents[1]
    files=[*sorted((root/'nice_validation').glob('*.py')),
           *sorted((root/'scripts').glob('*reevaluation*.py')),
           *sorted((root/'docs/reevaluation').glob('Dockerfile*')),
           root/'docs/reevaluation/vllm_policy.py']
    return {str(p.relative_to(root)):checksum(p) for p in files}


def resources(cfg,dep,old):
    inv=inventory(old)
    if inv['status']!='RESOURCES_PRESENT':
        raise ValueError('Old resources missing or changed: '+json.dumps(inv['issues']))
    snapshot=load(old/'config/snapshot.json')['config']
    scopes={scope:load(Path(record['manifest'])) for scope,record in inv['scopes'].items()}
    dev=load(old/'manifests/development_task.json')
    if isinstance(dev,list):
        if len(dev)!=1:raise ValueError('Expected one frozen development task')
        dev=dev[0]
    if dev['instance_id'] in {t['instance_id'] for tasks in scopes.values() for t in tasks}:
        raise ValueError('Development task overlaps formal task IDs')
    old_map_path=Path(snapshot['agent']['image_map'])
    old_map=load(old_map_path)
    if load(old_map_path.with_name('task_images.lock.json'))['image_map_sha256']!=checksum(old_map_path):
        raise ValueError('Development task image map lock changed')
    entry=old_map[dev['instance_id']]
    if entry.get('public_environment_only') is not True or entry['base_commit']!=dev['base_commit']:
        raise ValueError('Development image is not the original public clean environment')
    agent_id=inspect_image(entry['image'])['Id']
    grader_id=inspect_image(entry.get('grader_image',entry['image']))['Id']
    if grader_id!=inspect_image(entry['official_image_name'])['Id']:
        raise ValueError('Development official grader alias changed')
    inv['scopes']['development']={'sha256':checksum(old/'manifests/development_task.json'),
        'tasks':[{'task_id':dev['instance_id'],'entry':entry,'agent_image_id':agent_id,
                  'grader_image_id':grader_id}]}
    dep=dict(dep,development_task=dev)
    for record in [t for s in inv['scopes'].values() for t in s['tasks']]:
        prepared=dep['task_images'][record['task_id']]
        reference=prepared['image']
        if '@sha256:' not in reference and not reference.startswith('sha256:'):
            raise ValueError('Derived task image must be pinned by digest')
        image=inspect_image(reference)
        if (prepared['parent_image_id']!=record['agent_image_id'] or
            image.get('Config',{}).get('Labels',{}).get('nice.parent_image_id')!=record['agent_image_id']):
            raise ValueError('Derived task image has wrong parent: '+record['task_id'])
        prepared['resolved_image_id']=image['Id']
    for key in ['sdk_image','vllm_image']:
        ref=dep[key]
        if '@sha256:' not in ref and not ref.startswith('sha256:'):
            raise ValueError('Pin prepared service images by digest')
        dep[key+'_resolved_id']=inspect_image(ref)['Id']
    policy=Path(dep['egress_policy_evidence'])
    dep['egress_policy_sha256']=checksum(policy)
    if dep['docker_network'] in {'host','none','bridge',''}:
        raise ValueError('A dedicated restricted Docker network is required')
    from nice_validation.reevaluation_server import command
    net=json.loads(command(['docker','network','inspect',dep['docker_network']]).stdout)[0]
    if net.get('Internal') is not True:
        raise ValueError('Task network must be internal=true; do not allow benchmark containers public egress')
    dep['network_id']=net['Id']
    address=ipaddress.ip_address(dep['gateway_bind'])
    if address.version!=4 or not address.is_private or address.is_loopback or address.is_unspecified:
        raise ValueError('Gateway bind must be a private Docker-reachable unicast address')
    from urllib.parse import urlsplit
    endpoint=urlsplit(dep['gateway_url'])
    if endpoint.scheme!='http' or endpoint.hostname!=str(address) or endpoint.port or endpoint.path not in {'','/'}:
        raise ValueError('gateway_url must be http://gateway_bind without port/path')
    if not Path(dep['reference_python']).is_file() or not Path(dep['grading_python']).is_file():
        raise ValueError('Existing reference/grading interpreters must be local files')
    services=cfg['resources']['gpu_count']//cfg['resources']['tensor_parallel_size']
    for key in ['gateway_port_start','upstream_port_start']:
        if type(dep[key]) is not int or not 1024<=dep[key]<=65536-services:
            raise ValueError('Invalid unprivileged service port range: '+key)
    dep['driver_evidence']=command(['nvidia-smi','--query-gpu=uuid,driver_version,name,memory.total',
                                    '--format=csv,noheader']).stdout
    return inv,scopes,snapshot,dep


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','preflight','report'])
    parser.add_argument('--config',type=Path,default=Path('configs/reevaluation.yaml'))
    parser.add_argument('--deployment',type=Path)
    parser.add_argument('--old-run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mock',action='store_true')
    parser.add_argument('--execute-server',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'));validate(cfg)
    if args.old_run:independent_paths(args.old_run,args.output)
    if not args.mock and (not args.execute_server or platform.system()!='Linux'):
        parser.error('Real execution requires authorized Linux server --execute-server')
    with exclusive(args.output):
        if args.mock:
            scopes={scope:[{'instance_id':f'{scope}-{i}'} for i in range(count)]
                    for scope,count in cfg['scopes'].items()}
            frozen={'synthetic':True,'config':cfg,'source':source_identity(),'scopes':scopes}
            journal=Journal(args.output,frozen)
            jobs=jobs_for(cfg['models'],scopes)
            if args.action!='report':schedule(journal,jobs,[MockBackend() for _ in range(8)])
            if args.action!='preflight':report(journal,jobs)
            print('SYNTHETIC ONLY: '+str(len(jobs))+' episodes; no real model results')
            return
        if not args.old_run or not args.deployment:
            parser.error('--old-run and machine-local --deployment are required')
        dep=load(args.deployment)
        inv,scopes,oldcfg,dep=resources(cfg,dep,args.old_run)
        dep['run_root']=str(args.output.resolve())
        journal=Journal(args.output,{'synthetic':False,'config':cfg,'source':source_identity(),
                         'inventory':inv,'deployment':dep,'scopes':scopes})
        jobs=jobs_for(cfg['models'],scopes)
        if len(jobs)!=600:raise ValueError('Formal workload must be exactly 600 episodes')
        if args.action=='report':report(journal,jobs);return
        if args.action=='run' and all(journal.completed(job) is not None for job in jobs):
            report(journal,jobs)
            print('All 600 episodes already complete; verified without starting GPU services')
            return
        from nice_validation.reevaluation_server import Services,ServerBackend,cleanup_orphans
        from nice_validation.reevaluation_acceptance import hf_reference,accept
        gate=args.output/'acceptance/acceptance.json'
        reuse=False
        if gate.exists():
            receipt=load(gate)
            reuse=receipt.get('status')=='PASS' and receipt.get('run_id')==journal.run_id
            if not reuse:raise ValueError('Acceptance belongs to a different frozen run')
            for name,sha in receipt['artifacts'].items():
                if checksum(gate.parent/name)!=sha:raise ValueError('Acceptance artifact changed')
        cleanup_orphans(args.output)
        reference=None if reuse else hf_reference(cfg,dep,inv,args.output/'reference')
        tasks={(scope,t['instance_id']):t for scope,ts in scopes.items() for t in ts}
        tasks[('development',dep['development_task']['instance_id'])]=dep['development_task']
        with Services(cfg,dep,inv,args.output) as services:
            backends=[ServerBackend(cfg,dep,inv,gateway,index,tasks,oldcfg)
                      for index,gateway in enumerate(services.gateways)]
            if not reuse:
                # Preserve failed acceptance evidence; next attempt gets its own directory.
                count=len(list(args.output.glob('acceptance-attempt-*')))
                acceptance_root=args.output/f'acceptance-attempt-{count:03d}'
                accept(cfg,dep,inv,services,backends,reference,acceptance_root,journal.run_id)
                save(gate,{'status':'PASS','run_id':journal.run_id,
                    'artifacts':{str(p.relative_to(gate.parent)):checksum(p) for p in acceptance_root.rglob('*') if p.is_file()}})
            if args.action=='run':
                schedule(journal,jobs,backends,cfg['agent']['infrastructure_retries'],pipeline=True)
                report(journal,jobs)
        print('Acceptance complete'+('; 600 episodes packaged' if args.action=='run' else '; no formal task executed'))


if __name__=='__main__':main()
