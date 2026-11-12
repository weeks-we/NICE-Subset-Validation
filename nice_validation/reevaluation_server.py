"""Server-only local vLLM services and SDK task containers. No pull/build/download."""
from __future__ import annotations
from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid

from .reevaluation_deploy import network_environment, inspect_image
from .reevaluation_gateway import Gateway, http_json
from .reevaluation_runtime import InfrastructureError, save, load


def command(args, timeout=60, **kwargs):
    try:
        return subprocess.run(args, check=True, text=True, capture_output=True, timeout=timeout, **kwargs)
    except (subprocess.SubprocessError, OSError) as exc:
        raise InfrastructureError(str(exc)) from exc


def cleanup_orphans(root):
    """Only this output root's labeled containers; never prune shared resources."""
    import docker
    with docker.from_env() as client:
        for container in client.containers.list(all=True,filters={'label':'nice.run_root='+str(Path(root).resolve())}):
            if not container.name.startswith('nice-reeval-'):
                raise ValueError('Unexpected container carries this run label; refusing cleanup')
            save(Path(root)/'orphan-cleanup'/f'{container.id}.json',
                 {'container_id':container.id,'name':container.name,'action':'remove_own_interrupted_container'})
            container.remove(force=True)


class Services:
    def __init__(self, cfg, deployment, inventory, root):
        self.cfg, self.dep, self.inv, self.root = cfg, deployment, inventory, Path(root)
        self.names, self.gateways, self.upstreams = [], [], []

    def __enter__(self):
        import docker
        self.client = docker.from_env()
        try:
            self._start()
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def _start(self):
        import docker
        cfg, dep = self.cfg, self.dep
        ids = dep['gpu_uuids']
        if len(ids) != cfg['resources']['gpu_count'] or len(set(ids)) != len(ids):
            raise ValueError('Explicit unique authorized GPU UUIDs must match gpu_count')
        available = command(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).stdout.split()
        if not set(ids).issubset(available) or any(not re.fullmatch(r'GPU-[0-9a-fA-F-]+', x) for x in ids):
            raise ValueError('Authorized GPU UUID is not present')
        tp = cfg['resources']['tensor_parallel_size']
        if tp not in {1,2} or len(ids) % tp:
            raise ValueError('Only pre-frozen TP1/TP2 service groups are supported')
        image = dep['vllm_image']
        if '@sha256:' not in image and not image.startswith('sha256:'):
            raise ValueError('Freeze the prepared vLLM policy image by digest')
        self.client.images.get(image)  # Local only, never pull.
        mounts = {self.inv['base_path']:{'bind':'/model','mode':'ro'}}
        for model, record in self.inv['adapters'].items():
            mounts[record['path']] = {'bind':'/adapters/'+model,'mode':'ro'}
        for index, start in enumerate(range(0,len(ids),tp)):
            port = dep['upstream_port_start']+index
            name = 'nice-reeval-vllm-'+uuid.uuid4().hex[:12]
            args = ['--model','/model','--served-model-name','base_reference',
                '--tokenizer','/model','--dtype','bfloat16','--max-model-len',str(cfg['generation']['context_length']),
                '--tensor-parallel-size',str(tp),'--gpu-memory-utilization','0.90',
                '--enable-lora','--max-lora-rank','16','--max-loras','8','--max-cpu-loras','8',
                '--enable-auto-tool-choice','--tool-call-parser','qwen3_coder',
                '--reasoning-parser','qwen3','--port','8000','--lora-modules']
            args += [m+'=/adapters/'+m for m in cfg['models']]
            container = self.client.containers.run(image, args, name=name, detach=True,
                volumes=mounts, ports={'8000/tcp':('127.0.0.1',port)},
                device_requests=[docker.types.DeviceRequest(device_ids=ids[start:start+tp],capabilities=[['gpu']])],
                shm_size='16g', environment={'NICE_GENERATION_POLICY':json.dumps(cfg['generation']),
                    'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'},
                labels={'nice.reevaluation':'true','nice.run_root':str(self.root.resolve())})
            self.names.append(container)
            expected = __import__('hashlib').sha256((Path(__file__).resolve().parents[1]/'docs/reevaluation/vllm_policy.py').read_bytes()).hexdigest()
            check = container.exec_run(['python','-c',
                'import hashlib;print(hashlib.sha256(open("/opt/nice-vllm-policy.py","rb").read()).hexdigest())'])
            if check.exit_code or check.output.decode().strip()!=expected:
                raise InfrastructureError('Prepared vLLM image does not contain the frozen generation policy')
            upstream = f'http://127.0.0.1:{port}'
            deadline = time.monotonic()+dep.get('service_start_seconds',900)
            while True:
                container.reload()
                if container.status in {'dead','exited'}:
                    raise InfrastructureError('vLLM startup failed: '+container.logs(tail=80).decode(errors='replace'))
                try:
                    http_json(upstream+'/health',timeout=2)
                    break
                except Exception:
                    if time.monotonic()>deadline:
                        raise InfrastructureError('vLLM startup timeout')
                    time.sleep(1)
            if 'NICE_EFFECTIVE_GENERATION_POLICY=' not in container.logs().decode(errors='replace'):
                raise InfrastructureError('vLLM policy entrypoint was not used')
            model_cards = http_json(upstream+'/v1/models')['data']
            aliases = {m['id'] for m in model_cards}
            if not set(cfg['models']).issubset(aliases):
                raise InfrastructureError('Missing static LoRA model aliases')
            for model in cfg['models']:
                if next(m for m in model_cards if m['id']==model).get('root')!='/adapters/'+model:
                    raise InfrastructureError('LoRA alias points at an unexpected adapter path')
            gateway = Gateway((dep['gateway_bind'],dep['gateway_port_start']+index), upstream,cfg).start()
            self.upstreams.append(upstream)
            self.gateways.append(gateway)

    def __exit__(self, *unused):
        for gateway in self.gateways:
            gateway.close()
        for container in self.names:
            try:
                log = container.logs().decode(errors='replace')
                (self.root/(container.name+'.log')).write_text(log,encoding='utf-8')
            finally:
                container.remove(force=True)


class ServerBackend:
    def __init__(self, cfg, deployment, inventory, gateway, index, tasks, old_config):
        self.cfg, self.dep, self.inv, self.gateway = cfg, deployment, inventory, gateway
        self.index, self.tasks, self.old_config = index, tasks, old_config
        self.cancelled=threading.Event()

    def cancel(self):
        self.cancelled.set()

    def agent(self, job, attempt):
        if self.cancelled.is_set():
            raise RuntimeError('Controller stopped; preserve attempt for task-level recovery')
        from .harness import patch_command
        task = self.tasks[(job['scope'],job['task_id'])]
        entry = self.dep['task_images'][job['task_id']]
        endpoint = self.dep['gateway_url'].rstrip('/')+':'+str(self.dep['gateway_port_start']+self.index)
        token, budget = self.gateway.register(job['model'],attempt/'budget.json')
        env = network_environment(endpoint)
        env.update(ANTHROPIC_API_KEY=token, ANTHROPIC_AUTH_TOKEN=token,
            CLAUDE_CODE_MAX_OUTPUT_TOKENS=str(self.cfg['generation']['max_new_tokens']),
            CLAUDE_CODE_AUTO_COMPACT_WINDOW=str(self.cfg['generation']['context_length']),
            CLAUDE_AUTOCOMPACT_PCT_OVERRIDE='80',
            BASH_MAX_TIMEOUT_MS=str(self.cfg['agent']['tool_seconds']*1000),
            BASH_DEFAULT_TIMEOUT_MS=str(self.cfg['agent']['tool_seconds']*1000))
        name = 'nice-reeval-task-'+uuid.uuid4().hex[:12]
        root = self.old_config['agent']['workdir']
        image = entry['image']
        original = next(t for s in self.inv['scopes'].values() for t in s['tasks'] if t['task_id']==job['task_id'])
        if entry['parent_image_id'] != original['agent_image_id']:
            raise ValueError('Derived task image parent does not match original')
        info = inspect_image(image)
        if info.get('Config',{}).get('Labels',{}).get('nice.parent_image_id') != original['agent_image_id']:
            raise ValueError('Derived task image lacks frozen parent identity label')
        runtime = '/opt/nice-sdk/bin/python'
        command(['docker','create','--name',name,'--pull=never','--network',self.dep['docker_network'],
                 '--label','nice.run_root='+self.dep['run_root'],
                 '--cap-drop=ALL','--cap-add=CHOWN','--cap-add=FOWNER',
                 '--cap-add=DAC_OVERRIDE','--cap-add=KILL',
                 '--security-opt=no-new-privileges','--pids-limit','512',
                 '--entrypoint','/bin/bash',image,'-c','sleep infinity'])
        try:
            command(['docker','start',name])
            commit=original['entry'].get('checkout_commit') or task['base_commit']
            actual=command(['docker','exec','--user','0',name,'git','-c','safe.directory='+root,'-C',root,'rev-parse','HEAD']).stdout.strip()
            if actual!=commit:
                raise InfrastructureError('Task checkout differs from frozen clean starting commit')
            dirty=command(['docker','exec','--user','0',name,'git','-c','safe.directory='+root,'-C',root,'status','--porcelain','--untracked-files=all']).stdout.strip()
            hidden=command(['docker','exec','--user','0',name,'bash','-c',
                'for p in /eval.sh /tmp/test.patch /tmp/patch.diff; do if [ -e "$p" ]; then echo "$p"; fi; done']).stdout.strip()
            if dirty or hidden:
                raise InfrastructureError('Task SDK image contains dirty checkout or grader artifacts')
            # Reference answers and grading scripts are never copied into the SDK workspace.
            setup = ('chown -R 1000:1000 '+shlex.quote(root)+
                     ' && mkdir -p /tmp/sdk-home && chown 1000:1000 /tmp/sdk-home')
            command(['docker','exec','--user','0',name,'bash','-lc',setup],timeout=180)
            shell_setup = original['entry'].get('shell_setup','')
            command(['docker','exec','--user','0',name,'bash','-c',
                     'printf %s '+shlex.quote(shell_setup+'\n')+' > /tmp/sdk-home/bash-env; chmod 644 /tmp/sdk-home/bash-env'])
            env['BASH_ENV'] = '/tmp/sdk-home/bash-env'
            source = Path(__file__).with_name('sdk_agent.py').read_text(encoding='utf-8')
            prompt = ('Fix the repository issue below. Inspect, edit and test the repository. '
                      'Finish when the patch is ready; do not commit.\n\n'+task['problem_statement'])
            if job.get('probe') == 'context':
                # Engineering stress input only, never a benchmark prompt or shortened context.
                prompt = ('Context continuity acceptance. Use Bash to print the word marker. '
                          'Then use Bash to print another marker. Keep doing this for 12 tool calls, '
                          'then finish. Preserve the task instruction after any automatic compaction.\n'
                          + ' x'*220000)
            invocation = '\nimport asyncio\n'
            invocation += 'try:\n r=asyncio.run(sdk_session(**'+repr({'prompt':prompt,'model':job['model'],
                'workdir':root,'env':env,'cli_path':self.cfg['claude_runtime_path'],
                'events_path':'/tmp/sdk-home/events.jsonl','max_turns':self.cfg['agent']['max_turns'],
                'wall_seconds':self.cfg['agent']['wall_seconds'],'tool_seconds':self.cfg['agent']['tool_seconds'],
                'allow_action_limit':True})+',event_sink=lambda line:print(line,flush=True)))\n'
            invocation += ' print(json.dumps({"sdk_type":"SessionReceipt","data":r}),flush=True)\n'
            invocation += 'except Exception as e:\n print(json.dumps({"sdk_type":"SessionError","data":{"error":str(e)}}),flush=True)\n raise\n'
            worker = attempt/'worker.py'
            worker.write_text(source+invocation,encoding='utf-8')
            command(['docker','cp',str(worker),name+':/tmp/sdk-home/worker.py'])
            # Remove credentials from persisted source; the live copy is confined to this task container.
            worker.write_text('Credentials redacted; worker source is frozen in sdk_agent.py.\n',encoding='utf-8')
            args = ['docker','exec','--user','1000:1000','--workdir',root,'--env','HOME=/tmp/sdk-home',
                    name,runtime,'/tmp/sdk-home/worker.py']
            records, active, final, error = [], {}, None, None
            events = queue.Queue()
            stderr = (attempt/'sdk.stderr.log').open('w',encoding='utf-8')
            process = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
            def drain():
                for line in process.stdout:
                    events.put(line)
                events.put(None)
            reader = threading.Thread(target=drain,daemon=True); reader.start()
            stop = None
            with (attempt/'sdk.events.jsonl').open('x',encoding='utf-8') as log:
                try:
                    while True:
                        now = time.monotonic()
                        if self.cancelled.is_set():
                            stop='CONTROLLER_INTERRUPTED';break
                        if now-budget.started >= self.cfg['agent']['wall_seconds']:
                            stop = 'WALL_LIMIT'; break
                        if any(now-t > self.cfg['agent']['tool_seconds'] for t in active.values()):
                            stop = 'TOOL_TIMEOUT'; break
                        if budget.stop_reason:
                            stop = budget.stop_reason; break
                        try:
                            line = events.get(timeout=.2)
                        except queue.Empty:
                            continue
                        if line is None:
                            break
                        log.write(line); log.flush()
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        records.append(event)
                        data = event.get('data',{})
                        if event.get('sdk_type') == 'ToolSupervisor':
                            if data.get('event') == 'PreToolUse':
                                active[data['tool_use_id']] = now
                            else:
                                active.pop(data.get('tool_use_id'),None)
                        elif event.get('sdk_type') == 'SessionReceipt': final = data
                        elif event.get('sdk_type') == 'SessionError': error = data
                finally:
                    if stop:
                        budget.stop_reason=stop
                    budget.cancel.set()
                    # Kill every agent-owned process, including detached Bash children, before taking the diff.
                    killer = 'import os,signal\nfor p in os.listdir("/proc"):\n try:\n  if p.isdigit() and os.stat("/proc/"+p).st_uid==1000:os.kill(int(p),signal.SIGKILL)\n except (FileNotFoundError,ProcessLookupError,PermissionError):pass'
                    command(['docker','exec','--user','0',name,runtime,'-c',killer])
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                    reader.join(timeout=5); stderr.close()
                    if not budget.lock.acquire(timeout=30):
                        raise ValueError('Upstream generation did not stop after task cancellation; restart services before continuing')
                    budget.lock.release()
            patch = command(['docker','exec','--user','1000:1000',name,'bash','-lc',patch_command(root,commit)],timeout=120).stdout
            (attempt/'patch.diff').write_text(patch,encoding='utf-8')
            if stop=='CONTROLLER_INTERRUPTED':
                raise RuntimeError('Controller interrupted this infrastructure attempt; no model result published')
            if stop == 'INFRASTRUCTURE_ERROR' or (not stop and not final):
                raise InfrastructureError('SDK/API failed without a valid model outcome: '+str(error))
            if not stop and budget.requests<1:
                raise InfrastructureError('SDK reported success without any accounted inference request')
            stop = stop or ('ACTION_LIMIT' if final['result']['subtype']=='error_max_turns' else 'SUBMITTED')
            budget.persist()
            return {'patch':patch,'stop_reason':stop,'usage':{
                'input_tokens':budget.input_tokens,'output_tokens':budget.output_tokens,
                'incomplete_usage':budget.incomplete_usage},
                'compactions':sum(e.get('sdk_type')=='SystemMessage' and e.get('data',{}).get('subtype')=='compact_boundary' for e in records),
                'thinking_blocks':sum(e.get('sdk_type')=='AssistantMessage' and any(b.get('thinking') for b in e.get('data',{}).get('content',[])) for e in records)}
        finally:
            self.gateway.sessions.pop(token,None)
            command(['docker','rm','-f',name])

    def grade(self, job, patch, attempt):
        cfg = deepcopy(self.old_config)
        scope = 'benchmark' if job['scope']=='benchmark' else 'nice'
        if job['scope']!='development':
            cfg['agent']['image_map'] = str(Path(self.inv['old_run'])/'environments'/scope/'task_images.json')
        cfg['agent']['evaluation_timeout'] = self.cfg['agent']['grading_seconds']
        kind = 'swesmith' if job['scope']=='development' else 'swebench'
        cfg['data']['validation_kind'] = kind
        cfg.setdefault('transfer',{})['enabled'] = False
        task = dict(self.tasks[(job['scope'],job['task_id'])],evaluation_backend=kind)
        patch_path = attempt/'patch.diff'; patch_path.write_text(patch,encoding='utf-8')
        request = attempt/'grade_request.json'
        save(request, {'config':cfg,'task':task,'patch':str(patch_path.resolve()),
                       'output':str((attempt/'grade').resolve())})
        script = Path(__file__).resolve().parents[1]/'scripts/reevaluation_grade.py'
        try:
            result = command([self.dep['grading_python'],str(script),str(request.resolve())],
                             timeout=self.cfg['agent']['grading_seconds']+600)
        except InfrastructureError as exc:
            cause=exc.__cause__
            for name in ['stdout','stderr']:
                value=getattr(cause,name,'') or ''
                if isinstance(value,bytes):value=value.decode(errors='replace')
                (attempt/('grade.'+name+'.log')).write_text(value,encoding='utf-8')
            raise
        (attempt/'grade.stdout.log').write_text(result.stdout,encoding='utf-8')
        (attempt/'grade.stderr.log').write_text(result.stderr,encoding='utf-8')
        verdict = load(attempt/'grade/verdict.json')
        return verdict['resolved']
