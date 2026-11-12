"""Pinned vLLM entrypoint: enforce settings native Anthropic conversion omits."""
import importlib.metadata
import json
import os
import runpy


def install(policy):
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
    original = AnthropicServingMessages._build_base_request.__func__
    def convert(cls, request, messages):
        result = original(cls, request, messages)
        result.chat_template_kwargs = {'enable_thinking':policy['thinking'],
            'preserve_thinking':policy['preserve_thinking'], 'reasoning_effort':policy['reasoning_effort']}
        for name in ['temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty']:
            setattr(result, name, policy[name])
        return result
    AnthropicServingMessages._build_base_request = classmethod(convert)


if __name__ == '__main__':
    if importlib.metadata.version('vllm') != '0.22.0':
        raise RuntimeError('vLLM version differs from audited Anthropic conversion')
    policy = json.loads(os.environ['NICE_GENERATION_POLICY'])
    install(policy)
    print('NICE_EFFECTIVE_GENERATION_POLICY='+json.dumps(policy,sort_keys=True), flush=True)
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
