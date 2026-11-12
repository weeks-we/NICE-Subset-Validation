"""Run with the existing, source-locked official grader Python environment."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nice_validation.harness import official_grade

if __name__ == '__main__':
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    resolved = official_grade(job['config'], job['task'],
                              Path(job['patch']).read_text(encoding='utf-8'), Path(job['output']))
    print(json.dumps({'resolved':resolved}))
