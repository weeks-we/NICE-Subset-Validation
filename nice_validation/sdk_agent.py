"""Native Agent SDK session; no training, model download or official grader."""
from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import asdict, is_dataclass
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from urllib.parse import urlsplit

SDK_VERSION = '0.2.87'
CLI_VERSION = '2.1.153'
TOOLS = ['Bash', 'Read', 'Edit', 'Write', 'Glob', 'Grep']


def sdk_options(model, workdir, env, cli_path, max_turns=250):
    from claude_agent_sdk import ClaudeAgentOptions
    if not model or '/' in model or any(c.isspace() for c in model):
        raise ValueError('Use a nonempty served adapter alias without slash')
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError('max_turns must be a positive integer')
    if not (Path(cli_path).is_absolute() or PurePosixPath(cli_path).is_absolute()):
        raise ValueError('Explicit absolute Claude executable path required')
    env = dict(env)
    url = urlsplit(env.get('ANTHROPIC_BASE_URL', ''))
    if (url.scheme not in {'http','https'} or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path not in {'','/'}):
        raise ValueError('Explicit internal model service root URL required; no cloud fallback')
    if not env.get('ANTHROPIC_API_KEY') or not env.get('ANTHROPIC_AUTH_TOKEN'):
        raise ValueError('Explicit local inference credentials required')
    env.pop('CLAUDECODE', None)
    env.update(ANTHROPIC_MODEL=model, CLAUDE_CODE_SUBAGENT_MODEL=model,
               DISABLE_AUTOUPDATER='1', CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',
               **{'ANTHROPIC_DEFAULT_'+tier+'_MODEL':model for tier in ['OPUS','SONNET','HAIKU']})
    return ClaudeAgentOptions(
        model=model, cwd=str(workdir), cli_path=str(cli_path),
        system_prompt={'type': 'preset', 'preset': 'claude_code'},
        tools=list(TOOLS), allowed_tools=list(TOOLS),
        disallowed_tools=['Task', 'Agent', 'AskUserQuestion', 'WebSearch', 'WebFetch'],
        permission_mode='bypassPermissions', max_turns=max_turns,
        setting_sources=[], mcp_servers={}, strict_mcp_config=True, env=env,
        max_buffer_size=16 * 1024 * 1024,
    )


async def sdk_session(prompt, model, workdir, env, cli_path, events_path,
                      max_turns=250, wall_seconds=7200, event_sink=None,
                      tool_seconds=None, allow_action_limit=False):
    """Write every SDK event immediately; an interrupted file is NOT completion."""
    from claude_agent_sdk import query
    if importlib.metadata.version('claude-agent-sdk') != SDK_VERSION:
        raise ValueError('Agent SDK version mismatch')
    version = subprocess.check_output([str(cli_path), '--version'], text=True, timeout=30).strip()
    if not version or version.split()[0] != CLI_VERSION:
        raise ValueError('Claude runtime version mismatch: ' + version)
    if wall_seconds <= 0:
        raise ValueError('Positive SDK session wall timeout required')
    options = sdk_options(model, workdir, env, cli_path, max_turns)
    if tool_seconds is not None:
        from claude_agent_sdk import HookMatcher
        if tool_seconds <= 0:
            raise ValueError('Positive tool timeout required')
        async def hook(data, tool_use_id, context):
            event = {'sdk_type':'ToolSupervisor', 'data':{
                'event':data.get('hook_event_name'), 'tool_use_id':tool_use_id,
                'time':__import__('time').time(), 'tool_name':data.get('tool_name')}}
            if event_sink is not None:
                event_sink(json.dumps(event))
            if data.get('hook_event_name') == 'PreToolUse' and data.get('tool_name') == 'Bash':
                updated = dict(data.get('tool_input', {}))
                updated['timeout'] = min(updated.get('timeout') or tool_seconds*1000, tool_seconds*1000)
                updated['run_in_background'] = False
                return {'hookSpecificOutput':{'hookEventName':'PreToolUse', 'updatedInput':updated}}
            return {}
        options.hooks = {name:[HookMatcher(hooks=[hook])] for name in
                         ['PreToolUse','PostToolUse','PostToolUseFailure']}
    events_path = Path(events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    rounds, thinking, calls, results, final = 0, 0, [], [], None
    # Exclusive creation prevents accidentally overwriting earlier session evidence.
    with events_path.open('x', encoding='utf-8') as stream:
        async def consume():
            nonlocal rounds, thinking, final
            async with aclosing(query(prompt=prompt, options=options)) as messages:
                async for message in messages:
                    name = type(message).__name__
                    record = asdict(message) if is_dataclass(message) else vars(message)
                    line = json.dumps({'sdk_type': name, 'data': record}, ensure_ascii=False)
                    stream.write(line + '\n')
                    stream.flush()
                    if event_sink is not None:
                        event_sink(line)
                    if name == 'AssistantMessage':
                        if record.get('error'):
                            raise RuntimeError('Assistant API error: ' + str(record['error']))
                        rounds += 1
                        for block in record.get('content', []):
                            if block.get('name') and block.get('id'):
                                calls.append({'id': block['id'], 'name': block['name']})
                            if block.get('thinking'):
                                thinking += 1
                    elif name == 'UserMessage' and isinstance(record.get('content'), list):
                        results.extend(b for b in record['content'] if 'tool_use_id' in b)
                    elif name == 'ResultMessage':
                        if final is not None:
                            raise RuntimeError('Multiple SDK final events')
                        final = record
        await asyncio.wait_for(consume(), timeout=wall_seconds)
    action_limit = (final or {}).get('subtype') == 'error_max_turns'
    if final is None or ((final.get('is_error') or final.get('subtype') != 'success')
                         and not (allow_action_limit and action_limit)):
        raise RuntimeError('SDK session did not complete successfully: ' + str(final))
    return {'sdk_version': SDK_VERSION, 'cli_version': version, 'model': model,
            'assistant_rounds': rounds, 'tool_calls': calls,
            'tool_results': results, 'thinking_blocks_observed': thinking,
            'result': final, 'scope': 'SDK agent session only; no official grade/capacity certification'}
