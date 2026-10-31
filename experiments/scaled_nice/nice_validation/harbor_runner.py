"""One isolated native task attempt, with pristine Pro verifier replay."""

import asyncio
from pathlib import Path
import sys

from nice_validation.core import read


async def execute(job_file: Path) -> None:
    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    job = await Job.create(JobConfig.model_validate(read(job_file)))
    await job.run()


if __name__ == "__main__":
    asyncio.run(execute(Path(sys.argv[1])))
