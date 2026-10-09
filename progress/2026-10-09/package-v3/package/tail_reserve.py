"""Protect later analysis/export within, rather than in addition to, total caps."""
import math


def pending_tail_reserve(request, stage):
    order=request['stage_order']
    if stage not in order: raise ValueError('TAIL_RESERVE_UNKNOWN_STAGE')
    reserved=request.get('accounting_reserves',{}).get('protected_tail_stages',[])
    if not isinstance(reserved,list) or len(set(reserved))!=len(reserved):
        raise ValueError('TAIL_RESERVE_INVALID_STAGE_LIST')
    if any(s not in order for s in reserved): raise ValueError('TAIL_RESERVE_UNBUDGETED_STAGE')
    result={'wall_seconds':0.,'complete_process_cpu_seconds':0.}
    for name in reserved:
        if order.index(name)<=order.index(stage): continue
        for key in result:
            value=request['stages'][name][key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('TAIL_RESERVE_INVALID_CAP:'+name+':'+key)
            result[key]+=value
    return result


def worker_wall_allowance(request, elapsed):
    export=request['stage_order'][-1]
    if export!='settlement_and_verified_export': raise ValueError('EXPORT_STAGE_MUST_BE_LAST')
    cleanup=request.get('accounting_reserves',{}).get('cross_system_wall_seconds',0.)
    return request['totals']['wall_seconds']-elapsed-request['stages'][export]['wall_seconds']-cleanup
