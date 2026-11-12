from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
import urllib.error
import zipfile

import pytest
import yaml
from nice_validation.reevaluation_runtime import (Journal,MockBackend,InfrastructureError,
    jobs_for,schedule,report,load,save,checksum,exclusive)
from nice_validation.reevaluation_gateway import Budget,Gateway


@pytest.fixture
def cfg():return yaml.safe_load(Path('configs/reevaluation.yaml').read_text(encoding='utf-8'))


def test_completed_skip_resume_and_tamper_rejected(tmp_path):
    journal=Journal(tmp_path,{'synthetic':True})
    job={'scope':'benchmark','model':'initial','task_id':'test'}
    backend=MockBackend()
    first=journal.execute(job,backend)
    class NoModel:
        def agent(*a):pytest.fail('Completed model reran')
    assert journal.execute(job,NoModel())==first
    (journal.folder(job)/first['attempt']/'patch.diff').write_text('changed')
    with pytest.raises(ValueError,match='artifact changed'):journal.completed(job)


def test_grading_only_resume_and_retry_bound(tmp_path):
    journal=Journal(tmp_path,{'synthetic':True})
    job={'scope':'nice_retest','model':'initial','task_id':'x'}
    class Backend(MockBackend):
        calls=0
        def agent(self,*args):self.calls+=1;return super().agent(*args)
        def grade(self,*args):raise InfrastructureError('grader unavailable')
    backend=Backend()
    with pytest.raises(InfrastructureError):journal.execute(job,backend)
    assert backend.calls==1
    with pytest.raises(ValueError,match='exhausted'):journal.execute(job,backend)
    assert backend.calls==1


def test_interrupted_agent_retries_bounded_and_no_model_failure_retry(tmp_path):
    journal=Journal(tmp_path,{'synthetic':True})
    job={'scope':'benchmark','model':'initial','task_id':'x'}
    folder=journal.folder(job);(folder/'attempt-000').mkdir(parents=True)
    class Backend(MockBackend):
        calls=0
        def agent(self,*args):self.calls+=1;raise InfrastructureError('connection')
    backend=Backend()
    with pytest.raises(InfrastructureError):journal.execute(job,backend)
    assert backend.calls==2  # Interrupted attempt already consumed one of three attempts.


def test_model_budget_failure_is_graded_once(tmp_path):
    journal=Journal(tmp_path,{'synthetic':True})
    class Backend(MockBackend):
        calls=0
        def agent(self,*args):
            self.calls+=1
            return {'patch':'valid nonempty patch','stop_reason':'TOKEN_LIMIT'}
        def grade(self,job,patch,attempt):assert patch;return True
    backend=Backend();job={'scope':'benchmark','model':'initial','task_id':'x'}
    assert journal.execute(job,backend)['resolved']
    assert journal.execute(job,backend)['stop_reason']=='TOKEN_LIMIT'
    assert backend.calls==1


def test_pipeline_never_double_books_inference_and_grading_overlaps(tmp_path):
    import time
    shared={'active':0,'max':0,'grading':False,'overlap':False}
    lock=threading.Lock()
    class Backend(MockBackend):
        def agent(self,*args):
            with lock:
                shared['active']+=1;shared['max']=max(shared['max'],shared['active'])
                shared['overlap']|=shared['grading']
            time.sleep(.02)
            result=super().agent(*args)
            with lock:shared['active']-=1
            return result
        def grade(self,*args):
            shared['grading']=True
            time.sleep(.08)
            result=super().grade(*args)
            shared['grading']=False
            return result
    journal=Journal(tmp_path,{'synthetic':True})
    jobs=jobs_for(['initial'],{'benchmark':[{'instance_id':str(i)} for i in range(4)]})
    schedule(journal,jobs,[Backend()],pipeline=True)
    assert shared['max']==1 and shared['overlap']


def test_full_600_mock_parallel_serial_resume_reports(tmp_path,cfg):
    scopes={s:[{'instance_id':f'{s}-{i}'} for i in range(n)] for s,n in cfg['scopes'].items()}
    jobs=jobs_for(cfg['models'],scopes)
    assert len(jobs)==600
    parallel=Journal(tmp_path/'parallel',{'synthetic':True,'config':cfg})
    schedule(parallel,jobs,[MockBackend() for _ in range(8)])
    hashes={str(p):checksum(p) for p in (parallel.root/'episodes').rglob('*') if p.is_file()}
    schedule(parallel,jobs,[MockBackend()])
    assert hashes=={str(p):checksum(p) for p in (parallel.root/'episodes').rglob('*') if p.is_file()}
    serial=Journal(tmp_path/'serial',{'synthetic':True,'config':cfg})
    schedule(serial,jobs,[MockBackend()])
    assert [(parallel.completed(j)['resolved'],parallel.completed(j)['stop_reason']) for j in jobs]==[
        (serial.completed(j)['resolved'],serial.completed(j)['stop_reason']) for j in jobs]
    result=report(parallel,jobs)
    assert len(result)==10 and sum(v['total'] for v in result.values())==600
    completion=load(parallel.root/'controller/completion.json')
    bundle=parallel.root/'controller'/completion['zip']
    assert checksum(bundle)==completion['sha256'] and completion['synthetic']
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and 'reports/resolved.png' in z.namelist()
    assert (parallel.root/'reports/resolved.png').read_bytes().startswith(b'\x89PNG')


def test_incomplete_package_and_changed_frozen_config_fail(tmp_path):
    journal=Journal(tmp_path,{'synthetic':True})
    with pytest.raises(ValueError,match='Incomplete'):report(journal,[{'x':'y'}])
    with pytest.raises(ValueError,match='changed'):Journal(tmp_path,{'synthetic':False})
    with pytest.raises(ValueError,match='overlapping'):jobs_for(['initial'],{'a':[{'instance_id':'x'}],'b':[{'instance_id':'x'}]})


def test_budget_actual_usage_not_requested_max_and_context_limit(tmp_path,cfg):
    cfg=deepcopy(cfg);cfg['agent']['total_output_tokens']=9
    budget=Budget('initial',cfg,tmp_path/'budget.json')
    body=budget.prepare({'model':'initial','max_tokens':100},10)
    assert body['max_tokens']==9
    budget.finish(10,4,body)
    assert budget.output_tokens==4
    assert budget.prepare({'model':'initial'},10)['max_tokens']==5
    with pytest.raises(ValueError,match='CONTEXT_LIMIT'):
        budget.prepare({'model':'initial'},cfg['generation']['context_length'])
    assert load(tmp_path/'budget.json')['output_tokens']==4


def test_gateway_stream_usage_forced_settings_and_route_isolation(tmp_path,cfg):
    seen=[]
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            self.send_response(200);self.end_headers()
            if self.path.endswith('count_tokens'):
                self.wfile.write(b'{"input_tokens": 15}')
            else:
                for event in [{'type':'message_start','message':{'usage':{'input_tokens':15}}},
                              {'type':'message_delta','usage':{'output_tokens':3}}, {'type':'message_stop'}]:
                    self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
    upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    thread=threading.Thread(target=upstream.serve_forever,daemon=True);thread.start()
    gateway=Gateway(('127.0.0.1',0),'http://127.0.0.1:'+str(upstream.server_port),cfg).start()
    token,budget=gateway.register('initial',tmp_path/'budget.json')
    def post(model,key=token):
        req=urllib.request.Request('http://127.0.0.1:'+str(gateway.server.server_port)+'/v1/messages',
            data=json.dumps({'model':model,'messages':[],'stream':True,'max_tokens':100,'temperature':0}).encode(),
            headers={'x-api-key':key})
        with urllib.request.urlopen(req,timeout=5) as response:return response.read()
    try:
        assert b'message_stop' in post('initial')
        with budget.lock:assert budget.output_tokens==3 and budget.input_tokens==15
        assert seen[-1]['temperature']==1 and seen[-1]['max_tokens']==100
        assert seen[-1]['chat_template_kwargs']['preserve_thinking']
        with pytest.raises(urllib.error.HTTPError):post('other')
        with pytest.raises(urllib.error.HTTPError):post('initial','wrong-key')
    finally:
        gateway.close();upstream.shutdown();upstream.server_close();thread.join()


def test_vllm_conversion_patch_forwards_omitted_sampling(monkeypatch,cfg):
    import sys
    from types import ModuleType,SimpleNamespace
    module=ModuleType('vllm.entrypoints.anthropic.serving')
    class Serving:
        @classmethod
        def _build_base_request(cls,request,messages):return SimpleNamespace()
    module.AnthropicServingMessages=Serving
    monkeypatch.setitem(sys.modules,'vllm.entrypoints.anthropic.serving',module)
    spec=importlib.util.spec_from_file_location('policy','docs/reevaluation/vllm_policy.py')
    policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    policy.install(cfg['generation'])
    result=Serving._build_base_request(None,[])
    assert result.min_p==0 and result.repetition_penalty==1
    assert result.chat_template_kwargs['reasoning_effort']=='xhigh'
