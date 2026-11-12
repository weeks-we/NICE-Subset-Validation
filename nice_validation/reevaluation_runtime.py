"""Independent SDK reevaluation journal and scheduler (never opens old runs for writing)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import zipfile


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def exclusive(root):
    """Kernel lock releases after a crash; the evidence file can remain safely."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'controller.lock').open('a+b') as f:
        f.seek(0)
        f.write(b'0')
        f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


class InfrastructureError(RuntimeError):
    """Only this class is eligible for bounded retries; model failures are results."""


class Journal:
    def __init__(self, root, frozen):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.frozen = frozen
        self.run_id = identity(frozen)
        lock = self.root / 'frozen.json'
        if lock.exists():
            if load(lock) != frozen:
                raise ValueError('Frozen reevaluation inputs changed; use a new output directory')
        else:
            if any(self.root.iterdir()):
                if set(p.name for p in self.root.iterdir()) != {'controller.lock'}:
                    raise ValueError('Output is not a fresh reevaluation directory')
            save(lock, frozen)

    def folder(self, job):
        return self.root / 'episodes' / identity(job)

    def completed(self, job):
        folder = self.folder(job)
        if not (folder / 'result.json').exists():
            return None
        result = load(folder / 'result.json')
        if result['job'] != job or result['run_id'] != self.run_id:
            raise ValueError('Episode identity mismatch')
        for name, expected in result['artifacts'].items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder.resolve()) or checksum(path) != expected:
                raise ValueError('Completed episode artifact changed: ' + name)
        return result

    def execute(self, job, backend, retries=2):
        previous = self.completed(job)
        if previous:
            return previous
        folder = self.folder(job)
        folder.mkdir(parents=True, exist_ok=True)
        attempts = sorted(folder.glob('attempt-*'))
        # A published agent result survives grader interruption: do not call the model again.
        pending = next((p for p in reversed(attempts) if (p / 'agent.json').exists()
                        and not (p / 'infrastructure_error.json').exists()), None)
        while True:
            attempt = pending or folder / f'attempt-{len(attempts):03d}'
            if not pending:
                if len(attempts) > retries:
                    raise InfrastructureError('Infrastructure retry budget exhausted: ' + str(job))
                attempt.mkdir()
                attempts.append(attempt)
                save(attempt / 'intent.json', {'job': job, 'started': time.time()})
            started = time.monotonic()
            try:
                if pending:
                    retry_file=attempt/'grade_retries.json'
                    if retry_file.exists() and len(load(retry_file))>retries:
                        raise ValueError('Grading retry budget already exhausted; manual infrastructure repair required')
                    agent = load(attempt / 'agent.json')
                else:
                    agent = backend.agent(job, attempt)
                    if not isinstance(agent.get('patch'), str) or not agent.get('stop_reason'):
                        raise ValueError('Backend did not publish an explicit model outcome')
                    save(attempt / 'agent.json', agent)
                resolved = backend.grade(job, agent['patch'], attempt)
                if type(resolved) is not bool:
                    raise ValueError('Official grader must return a boolean, not a truthy object')
                value = {'job': job, 'run_id': self.run_id, 'resolved': resolved,
                         'stop_reason': agent['stop_reason'], 'usage': agent.get('usage', {}),
                         'elapsed_seconds': time.monotonic() - started,
                         'attempt': attempt.name, 'synthetic': bool(self.frozen['synthetic'])}
                value['artifacts'] = {str(p.relative_to(folder)): checksum(p)
                                      for p in sorted(attempt.rglob('*')) if p.is_file()}
                save(folder / 'result.json', value)
                return value
            except InfrastructureError as exc:
                # Keep the successful agent for grading-only retries, across process restarts too.
                if (attempt / 'agent.json').exists():
                    errors = load(attempt / 'grade_retries.json') if (attempt / 'grade_retries.json').exists() else []
                    errors.append(str(exc))
                    save(attempt / 'grade_retries.json', errors)
                    if len(errors) > retries:
                        raise
                    pending = attempt
                else:
                    save(attempt / 'infrastructure_error.json', {'error': str(exc)})
                    pending = None


def jobs_for(models, scopes):
    jobs = []
    seen = set()
    for scope, tasks in scopes.items():
        ids = [t['instance_id'] for t in tasks]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('Duplicate or overlapping task IDs')
        seen.update(ids)
        for model in models:
            for task_id in ids:
                jobs.append({'scope': scope, 'model': model, 'task_id': task_id})
    if len(models) != len(set(models)):
        raise ValueError('Duplicate model aliases')
    return jobs


def schedule(journal, jobs, backends, retries=2, pipeline=False):
    """One sequential task stream per inference service, no accidental extra slots."""
    if not backends:
        raise ValueError('No inference services')
    import queue
    if pipeline:
        # Release an inference slot as soon as its patch is saved; grading can
        # overlap the next episode. Bound both GPU slots and grading concurrency.
        available=queue.Queue()
        for backend in backends:available.put(backend)
        grading=threading.Semaphore(len(backends))
        class PooledBackend:
            def agent(self,job,attempt):
                backend=available.get()
                try:return backend.agent(job,attempt)
                finally:available.put(backend)
            def grade(self,job,patch,attempt):
                with grading:return backends[0].grade(job,patch,attempt)
        workers=[PooledBackend() for _ in range(2*len(backends))]
    else:
        workers=backends
    queue_ = queue.Queue()
    for job in jobs:
        if journal.completed(job) is None:
            queue_.put(job)
    stop = threading.Event()
    def cancel():
        stop.set()
        for backend in backends:
            if hasattr(backend,'cancel'):backend.cancel()
    def worker(backend):
        while not stop.is_set():
            try:
                job = queue_.get_nowait()
            except queue.Empty:
                return
            try:
                journal.execute(job, backend, retries)
            except BaseException:
                cancel()
                raise
            finally:
                queue_.task_done()
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = [pool.submit(worker, b) for b in workers]
        try:
            for future in futures:
                future.result()
        except BaseException:
            cancel()
            raise


def report(journal, jobs):
    rows = [journal.completed(j) for j in jobs]
    if any(r is None for r in rows):
        raise ValueError('Incomplete run; final package is forbidden')
    actual = {p.parent.name for p in (journal.root / 'episodes').glob('*/result.json')}
    if actual != {identity(j) for j in jobs}:
        raise ValueError('Unexpected episode results')
    out = journal.root / 'reports'
    out.mkdir(exist_ok=True)
    fields = ['scope', 'model', 'task_id', 'resolved', 'stop_reason', 'input_tokens',
              'output_tokens', 'incomplete_usage', 'elapsed_seconds', 'attempt', 'synthetic']
    with (out / 'task_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row['job'], **{k: row[k] for k in ['resolved','stop_reason','elapsed_seconds','attempt','synthetic']},
                             'input_tokens':row['usage'].get('input_tokens',0),
                             'output_tokens':row['usage'].get('output_tokens',0),
                             'incomplete_usage':row['usage'].get('incomplete_usage',False)})
    summary = {}
    for row in rows:
        key = row['job']['scope'] + '/' + row['job']['model']
        count = summary.setdefault(key, {'resolved':0, 'total':0})
        count['total'] += 1
        count['resolved'] += int(row['resolved'])
    save(out / 'summary.json', {'synthetic':journal.frozen['synthetic'], 'groups':summary})
    save(out / 'audit.json', {'status':'PASS', 'episodes':len(rows), 'run_id':journal.run_id,
                              'synthetic':journal.frozen['synthetic']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    keys = list(summary)
    ax.barh(keys, [summary[k]['resolved']/summary[k]['total'] for k in keys])
    ax.set_xlim(0,1)
    ax.set_xlabel('Resolved fraction')
    ax.set_title('SYNTHETIC PIPELINE TEST — NOT MODEL RESULTS' if journal.frozen['synthetic'] else 'SDK reevaluation (scopes reported separately)')
    fig.tight_layout()
    fig.savefig(out / 'resolved.png', dpi=150)
    plt.close(fig)
    controller = journal.root / 'controller'
    controller.mkdir(exist_ok=True)
    bundle = controller / ('results-' + journal.run_id[:16] + '.zip')
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in [journal.root/'frozen.json', *sorted(out.iterdir())]:
            z.write(p, p.relative_to(journal.root))
        z.writestr('episode_evidence.json', json.dumps(rows, ensure_ascii=False))
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
    save(controller / 'completion.json', {'status':'COMPLETE', 'run_id':journal.run_id,
         'episodes':len(rows), 'synthetic':journal.frozen['synthetic'], 'zip':bundle.name,
         'sha256':checksum(bundle)})
    return summary


class MockBackend:
    def agent(self, job, attempt):
        code = int(identity(job)[:8], 16)
        patch = 'synthetic patch' if code % 3 else ''
        (attempt/'patch.diff').write_text(patch, encoding='utf-8')
        return {'patch':patch, 'stop_reason':'ACTION_LIMIT' if code % 4 == 0 else 'SUBMITTED',
                'usage':{'input_tokens':100, 'output_tokens':20}}

    def grade(self, job, patch, attempt):
        resolved = bool(patch) and int(identity(job)[8:12],16) % 2 == 0
        save(attempt/'verdict.json', {'resolved':resolved, 'synthetic':True})
        return resolved
