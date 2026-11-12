import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml
from nice_validation.reevaluation_gateway import Budget
from nice_validation.reevaluation_runtime import load,save,InfrastructureError
import nice_validation.reevaluation_server as server
import nice_validation.reevaluation_acceptance as acceptance


def test_sdk_task_bridge_uses_original_checkout_hides_gold_and_stops_children(monkeypatch,tmp_path):
    cfg=yaml.safe_load(Path('configs/reevaluation.yaml').read_text(encoding='utf-8'))
    task={'instance_id':'dev','base_commit':'a'*40,'problem_statement':'Fix the public bug','patch':'GOLD_SECRET'}
    old={'agent':{'workdir':'/testbed','image_map':'old-map.json'},'data':{},'transfer':{'enabled':True}}
    dep={'task_images':{'dev':{'image':'sha256:derived','parent_image_id':'sha256:original'}},
         'run_root':str(tmp_path),
         'gateway_url':'http://172.30.0.1','gateway_port_start':19000,'docker_network':'private',
         'grading_python':'existing-python'}
    inv={'old_run':'original','scopes':{'development':{'tasks':[
        {'task_id':'dev','agent_image_id':'sha256:original',
         'entry':{'checkout_commit':'b'*40,'shell_setup':'export ORIGINAL_TESTBED=1'}}]}}}
    class Gateway:
        sessions={}
        def register(self,model,path):
            b=Budget(model,cfg,path);b.requests=1
            self.sessions['private-token']=b
            return 'private-token',b
    gateway=Gateway()
    monkeypatch.setattr(server,'inspect_image',lambda x:{'Config':{'Labels':{'nice.parent_image_id':'sha256:original'}}})
    calls=[]
    def command(args,**kwargs):
        calls.append(args)
        if args[0]=='existing-python':
            request=load(args[2]);assert request['config']['data']['validation_kind']=='swesmith'
            assert request['config']['agent']['image_map']=='old-map.json'
            assert request['config']['transfer']['enabled'] is False
            save(Path(request['output'])/'verdict.json',{'resolved':True})
        elif args[:2]==['docker','cp']:
            source=Path(args[2]).read_text(encoding='utf-8')
            assert 'GOLD_SECRET' not in source and 'Fix the public bug' in source
        if 'rev-parse' in args:return SimpleNamespace(stdout='b'*40,stderr='')
        if args[-2:-1]==['-lc'] and 'git -C' in args[-1]:
            assert 'b'*40 in args[-1]
            return SimpleNamespace(stdout='diff --git a/x b/x\n',stderr='')
        return SimpleNamespace(stdout='',stderr='')
    monkeypatch.setattr(server,'command',command)
    events=[{'sdk_type':'AssistantMessage','data':{'content':[{'thinking':'reasoning'}]}},
            {'sdk_type':'SessionReceipt','data':{'result':{'subtype':'success'}}}]
    class Process:
        def __init__(self,*a,**kw):self.stdout=io.StringIO(''.join(json.dumps(e)+'\n' for e in events))
        def wait(self,**kw):return 0
    monkeypatch.setattr(server.subprocess,'Popen',Process)
    backend=server.ServerBackend(cfg,dep,inv,gateway,0,{('development','dev'):task},old)
    job={'scope':'development','model':'initial','task_id':'dev'}
    agent=backend.agent(job,tmp_path)
    assert agent['stop_reason']=='SUBMITTED' and agent['thinking_blocks']==1 and agent['patch']
    assert 'private-token' not in (tmp_path/'worker.py').read_text(encoding='utf-8')
    assert any('SIGKILL' in str(c) for c in calls)
    assert calls[-1][:3]==['docker','rm','-f']
    assert not gateway.sessions
    assert backend.grade(job,agent['patch'],tmp_path)


def test_numerics_checks_all_five_adapters_and_rejects_wrong_logits(monkeypatch,tmp_path):
    cfg=yaml.safe_load(Path('configs/reevaluation.yaml').read_text(encoding='utf-8'))
    reference=tmp_path/'reference.gz'
    rows=[{'model':m,'prompt_ids':[1,2],'logprobs':[-float(i) for i in range(25)]}
          for m in cfg['models'] for _ in range(3)]
    with gzip.open(reference,'wt',encoding='utf-8') as f:json.dump(rows,f)
    def request(url,body,**kwargs):
        assert body['return_tokens_as_token_ids'] and body['prompt']==[1,2]
        return {'choices':[{'logprobs':{'top_logprobs':[{f'token_id:{i}':-float(i)+.01 for i in range(20)}]}}]}
    monkeypatch.setattr(acceptance,'http_json',request)
    assert len(acceptance.numerics('http://private',reference,cfg))==15
    rows[0]['logprobs'][0]=10
    with gzip.open(reference,'wt',encoding='utf-8') as f:json.dump(rows,f)
    with pytest.raises(InfrastructureError,match='numerical'):acceptance.numerics('http://private',reference,cfg)


def test_capacity_requires_actual_full_length_not_just_server_configuration(monkeypatch):
    cfg=yaml.safe_load(Path('configs/reevaluation.yaml').read_text(encoding='utf-8'))
    def request(url,body,**kwargs):
        assert len(body['prompt'])==229376 and body['max_tokens']==32768 and body['ignore_eos']
        return {'usage':{'prompt_tokens':229376,'completion_tokens':1}}
    monkeypatch.setattr(acceptance,'http_json',request)
    with pytest.raises(InfrastructureError,match='actually exercised'):
        acceptance.capacity('http://private',cfg,3)


def test_capacity_distinct_prefixes_do_not_share_entire_kv_cache(monkeypatch):
    cfg=yaml.safe_load(Path('configs/reevaluation.yaml').read_text(encoding='utf-8'))
    prefixes=[]
    def request(url,body,**kwargs):
        assert len(body['prompt'])==229376 and body['prompt'][1:]==[3]*229375
        prefixes.append(body['prompt'][0])
        return {'usage':{'prompt_tokens':229376,'completion_tokens':32768}}
    monkeypatch.setattr(acceptance,'http_json',request)
    for prefix in [10,11,12,13]:acceptance.capacity('http://private',cfg,3,prefix)
    assert len(set(prefixes))==4


def test_orphan_cleanup_filters_this_output_root_and_never_prunes(monkeypatch,tmp_path):
    import sys
    removed=[]
    container=SimpleNamespace(name='nice-reeval-task-owned',id='owned-id',
        remove=lambda **kw:removed.append(kw))
    class Client:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        class containers:
            @staticmethod
            def list(**kwargs):
                assert kwargs=={'all':True,'filters':{'label':'nice.run_root='+str(tmp_path.resolve())}}
                return [container]
    monkeypatch.setitem(sys.modules,'docker',SimpleNamespace(from_env=lambda:Client()))
    server.cleanup_orphans(tmp_path)
    assert removed==[{'force':True}]
    assert load(tmp_path/'orphan-cleanup/owned-id.json')['container_id']=='owned-id'
    container.name='unrelated-service'
    with pytest.raises(ValueError,match='Unexpected'):server.cleanup_orphans(tmp_path)
    assert len(removed)==1
