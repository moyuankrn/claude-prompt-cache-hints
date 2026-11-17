"""Add Claude cache breakpoints to request payloads without sending a request.

The OpenAI-compatible helper returns a copy. The Anthropic helper mutates its
payload and returns whether it changed. Neither helper stores prompt content.
"""
import copy


def count_marks(value):
    if isinstance(value,dict):
        return int(isinstance(value.get('cache_control'),dict)) + sum(count_marks(v) for k,v in value.items() if k!='cache_control')
    if isinstance(value,list):return sum(map(count_marks,value))
    return 0


def text(message):
    content=message.get('content','')
    if isinstance(content,str):return content
    if isinstance(content,list):return next((p.get('text','') for p in content if p.get('type')=='text'),'')
    return ''


def control(ttl):
    return {'type':'ephemeral','ttl':'1h'} if ttl=='1h' else {'type':'ephemeral'}


def mark(message,ttl):
    if message.get('role') not in ('user','system'):return False
    content=message.get('content')
    if isinstance(content,str) and content.strip():
        message['content']=[{'type':'text','text':content,'cache_control':control(ttl)}]
        return True
    if isinstance(content,list):
        for part in reversed(content):
            if part.get('type')=='text' and part.get('text','').strip():
                part['cache_control']=control(ttl);return True
    return False


def openai(payload,config,active_text):
    """Mark stable prefixes in an OpenAI-compatible request for a Claude model.

    ``config['prompt_cache']`` controls the policy. The caller is responsible
    for confirming that its gateway accepts Anthropic ``cache_control`` blocks.
    """
    settings=config.get('prompt_cache',{})
    if not isinstance(settings,dict) or not settings.get('enabled') or 'claude' not in payload.get('model','').lower():
        return payload
    # Preserve explicit caller controls; never exceed the provider's four marks.
    if count_marks(payload):return payload
    result=copy.deepcopy(payload)
    messages=result['messages']
    starts=[i for i,m in enumerate(messages) if m.get('role')=='user' and text(m)==active_text]
    end=starts[-1] if starts else 1
    static=settings.get('static_ttl','5m')
    tail=settings.get('tail_ttl','5m')
    # Longer TTLs must come first.
    if tail=='1h':static='1h'
    if messages and messages[0].get('role')=='system':mark(messages[0],static)
    turns=[i for i,m in enumerate(messages[:end]) if m.get('role')=='user'
           and not text(m).startswith(('以下为工具','以下为上面工具'))]
    block=max(2,min(20,int(settings.get('partition_turns',10))))
    fixed=((len(turns)-1)//block)*block if turns else 0
    if fixed>0:mark(messages[turns[fixed-1]],static)
    for message in reversed(messages[1:end]):
        if mark(message,tail):break
    return result


def anthropic(payload, ttl='5m'):
    """Keep client breakpoints and use spare slots for stable tools and turns.

    The current user turn (including time, recall and new attachments) remains
    outside our added breakpoint. A prior turn is immutable in later requests.
    """
    if ttl not in ('5m','1h'):raise ValueError('cache ttl must be 5m or 1h')
    if 'claude' not in payload.get('model','').lower():return False
    # Claude Code supplies its own 5m points. Upgrade them as well so the
    # entire selected prefix has the requested lifetime.
    changed=_upgrade_to_hour(payload) if ttl=='1h' else False
    if count_marks(payload)>=4:return changed
    if ttl=='5m' and _has_hour_mark(payload):return False  # Preserve TTL order.
    if _split_system_at_runtime(payload):changed=True
    tools=payload.get('tools',[])
    if tools and not any(t.get('cache_control') for t in tools) and count_marks(payload)<4:
        tools[-1]['cache_control']=control(ttl);changed=True
    system=payload.get('system')
    system_volatile=(isinstance(system,str) and _volatile(system)) or (
        isinstance(system,list) and any(_volatile(p.get('text','')) for p in system if isinstance(p,dict)))
    if count_marks(payload)<4 and not _system_marked(system):
        if isinstance(system,str) and system.strip() and not _volatile(system):
            payload['system']=[{'type':'text','text':system,'cache_control':control(ttl)}];changed=True
        elif isinstance(system,list):
            # Cache the full consecutive fixed prefix; leave the first
            # timestamped block and everything after it outside this mark.
            fixed=None
            for part in system:
                if not isinstance(part,dict) or part.get('type')!='text' or _volatile(part.get('text','')):
                    break
                if part.get('text','').strip():fixed=part
            if fixed:
                fixed['cache_control']=control(ttl);changed=True
    # A changing system block precedes every message. Marking old messages
    # would rewrite that varying prefix on every request, increasing cost.
    if count_marks(payload)<4 and not system_volatile:
        messages=payload.get('messages',[])
        starts=[i for i,m in enumerate(messages) if m.get('role')=='user'
                and not any(p.get('type')=='tool_result' for p in m.get('content',[]) if isinstance(p,dict))]
        end=starts[-1] if starts else len(messages)
        if not any(count_marks(m) for m in messages[:end]):
            for message in reversed(messages[:end]):
                if _mark_history(message,ttl):changed=True;break
    return changed


def _upgrade_to_hour(value):
    changed=False
    if isinstance(value,dict):
        mark=value.get('cache_control')
        if isinstance(mark,dict) and mark.get('type')=='ephemeral' and mark.get('ttl','5m')=='5m':
            mark['ttl']='1h';changed=True
        for key,part in value.items():
            if key!='cache_control':changed=_upgrade_to_hour(part) or changed
    elif isinstance(value,list):
        for part in value:changed=_upgrade_to_hour(part) or changed
    return changed


def _has_hour_mark(value):
    if isinstance(value,dict):
        mark=value.get('cache_control')
        if isinstance(mark,dict) and mark.get('ttl')=='1h':return True
        return any(_has_hour_mark(v) for k,v in value.items() if k!='cache_control')
    if isinstance(value,list):return any(map(_has_hour_mark,value))
    return False


def _system_marked(system):
    return isinstance(system,list) and any(p.get('cache_control') for p in system if isinstance(p,dict))


def _volatile(value):
    import re
    # Dates quoted in fixed instructions are not runtime state. Only treat a
    # concrete, labelled timestamp line as a changing system section.
    return bool(re.search(r'(?m)^\s*(?:Current date|Current time|当前日期|当前时间|北京时间)[:：]?\s*'
                          r'(?:20\d\d-\d\d-\d\d|\d\d:\d\d)',value))


def _split_system_at_runtime(payload):
    """Split a concrete runtime timestamp from preceding fixed instructions.

    Adjacent text blocks preserve every character in the original prompt.
    Existing client marks are never relocated.
    """
    import re
    system=payload.get('system')
    if _system_marked(system):return False
    blocks=[{'type':'text','text':system}] if isinstance(system,str) else system
    if not isinstance(blocks,list):return False
    for index,part in enumerate(blocks):
        if not isinstance(part,dict) or part.get('type')!='text' or part.get('cache_control'):continue
        value=part.get('text','')
        match=re.search(r'(?m)^\s*(?:Current date|Current time|当前日期|当前时间|北京时间)[:：]?\s*(?:20\d\d-\d\d-\d\d|\d\d:\d\d)',value)
        if match and match.start()>0 and value[:match.start()].strip():
            left={**part,'text':value[:match.start()]}
            right={**part,'text':value[match.start():]}
            payload['system']=blocks[:index]+[left,right]+blocks[index+1:]
            return True
    return False


def _mark_history(message,ttl='5m'):
    content=message.get('content')
    if not isinstance(content,list):
        if isinstance(content,str) and content.strip() and message.get('role') in ('user','assistant'):
            message['content']=[{'type':'text','text':content,'cache_control':control(ttl)}]
            return True
        return False
    for part in reversed(content):
        if part.get('cache_control'):return False
        if part.get('type') in ('text','tool_result','tool_use'):
            part['cache_control']=control(ttl);return True
    return False


def layout(payload):
    """Return breakpoint counts without copying prompt content."""
    system=payload.get('system',[])
    messages=payload.get('messages',[])
    return {'tools':count_marks(payload.get('tools',[])),
            'system':count_marks(system),
            'messages':count_marks(messages),
            'system_volatile':(isinstance(system,str) and _volatile(system)) or
                (isinstance(system,list) and any(_volatile(p.get('text','')) for p in system if isinstance(p,dict)))}
