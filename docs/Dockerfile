# Build from repository root. Actual base image and lock are supplied by server agent.
# Base MUST contain Python 3.12+, Git, Docker CLI, required OS libraries and the
# already validated CUDA PyTorch/compilers. Pin BASE_IMAGE by registry digest.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
WORKDIR /opt/nice
COPY . /opt/nice
# Export numerical dependencies from the validated server environment, excluding
# this editable project and the two harnesses. Include validated CUDA extensions.
# Never commit credentials, wheels, weights or the generated lock to Git.
ARG RUNTIME_LOCK=deployment/runtime.lock.txt
ARG SWEBENCH_COMMIT
ARG SWESMITH_COMMIT
RUN python -c "import sys; assert sys.version_info >= (3,12)" \
 && python -m pip install -r "${RUNTIME_LOCK}" \
 && python -c "import re; assert re.fullmatch('[a-f0-9]{40}', '${SWEBENCH_COMMIT}'); assert re.fullmatch('[a-f0-9]{40}', '${SWESMITH_COMMIT}')" \
 && python -m pip install -c "${RUNTIME_LOCK}" "git+https://github.com/SWE-bench/SWE-bench.git@${SWEBENCH_COMMIT}" "swesmith[validate] @ git+https://github.com/SWE-bench/SWE-smith.git@${SWESMITH_COMMIT}" \
 && python -m pip install --no-deps --no-build-isolation -e . \
 && python -m pip check \
 && python -m pip freeze > /opt/nice-runtime-freeze.txt
# Real GPU acceptance happens on the compute host, not during docker build.
# Offline flags are set by run_server only for deployment.offline=true.
CMD ["python", "scripts/run_server.py", "--help"]
