"""独立共享显存围栏和只读性能观测；原公共入口保持不变。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def append(path,value):
    with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def install_sampling_observer(path):
    """同步只用于资源测量，不更改采样参数、动作或筛选结果。"""
    import torch
    from policies.diffusion_model_wrapper import DiffusionModelWrapper as Wrapper
    sample=Wrapper._sample_policy;predict=Wrapper.predict
    def observed_sample(self,policy,device,batch,anchors,active):
        start=time.monotonic();success=False;error=None
        try:
            if device.type=='cuda':torch.cuda.synchronize(device)
            result=sample(self,policy,device,batch,anchors,active)
            if device.type=='cuda':torch.cuda.synchronize(device)
            success=True;return result
        except Exception as exc:error=type(exc).__name__;raise
        finally:
            append(path,dict(kind='sampling',scene_id=str(self.scene_id),step=getattr(self,'_resource_frame',None),
                row_count=int(batch['image'].shape[0]),num_samples=self.num_samples,anchor_active=bool(active),
                success=success,error_type=error,seconds=time.monotonic()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0)))
    def observed_predict(self,frame,attack_intent=None):
        self._resource_frame=int(frame.step)
        return predict(self,frame,attack_intent=attack_intent)
    Wrapper._sample_policy=observed_sample;Wrapper.predict=observed_predict
    def restore():Wrapper._sample_policy=sample;Wrapper.predict=predict
    return restore


def install_fixture(path):
    from policies.llm_adversarial_planner import LLMAdversarialPlanner as Planner
    from resource_policy import fixture_response
    original=Planner._request_attack_plan
    def request(planner,prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);response=fixture_response(payload)
        planner.last_successful_model_name=None;planner.last_attempted_model_names=[];planner.last_model_failures=[]
        planner.last_trace['resource_fixture']=dict(enabled=True,physical_LLM_requests=0,not_performance_evidence=True)
        append(path,dict(request_id=payload['request_id'],source_step=payload['context']['source_step'],
            selected=response['ranking'][0],response=response,not_LLM=True))
        return '',response
    Planner._request_attack_plan=request
    return lambda:setattr(Planner,'_request_attack_plan',original)


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--resource-cap-gib',type=float,required=True)
    p.add_argument('--resource-reserve-gib',type=float,default=3.)
    p.add_argument('--resource-fixture',choices=('off','on'),default='off')
    p.add_argument('--resource-observe-sampling',choices=('off','on'),default='off')
    args,remaining=p.parse_known_args();out=Path(remaining[remaining.index('--output-dir')+1]);prepare='--prepare-only' in remaining
    if out.exists():raise FileExistsError('preserve_prior_probe_or_arm')
    sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_provider_compat_20261028')]
    from resource_policy import memory_fraction
    if prepare:
        # 真实prepare路径不初始化CUDA或发送HTTP。
        import run_controlled
        old=sys.argv
        try:sys.argv=[sys.argv[0]]+remaining;return run_controlled.main()
        finally:sys.argv=old
    hashes=json.loads((out.parent/'source_sha256.json').read_text())
    for f in HERE.glob('*.py'):
        if hashes.get(f.relative_to(ROOT).as_posix())!=hashlib.sha256(f.read_bytes()).hexdigest():raise ValueError('shared_resource_runtime_not_frozen')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('exactly_one_visible_GPU_required')
    total=torch.cuda.get_device_properties(0).total_memory
    fraction=memory_fraction(args.resource_cap_gib,total)
    free,_=torch.cuda.mem_get_info(0)
    if free<(args.resource_cap_gib+args.resource_reserve_gib)*2**30:raise RuntimeError('shared_memory_admission_changed_before_load')
    torch.cuda.set_per_process_memory_fraction(fraction,device=0);torch.cuda.reset_peak_memory_stats(0)
    samples=out.with_name(out.name+'_resource_memory.jsonl');events=out.with_name(out.name+'_sampling.jsonl');fixture=out.with_name(out.name+'_fixture.jsonl')
    for f in [samples,events,fixture]:
        if f.exists():raise FileExistsError('preserve_resource_observation')
    stop=threading.Event()
    def memory_loop():
        while not stop.is_set():
            f,t=torch.cuda.mem_get_info(0)
            append(samples,dict(time=time.time(),allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0),
                free_bytes=f,total_bytes=t,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0)))
            stop.wait(2.)
    thread=threading.Thread(target=memory_loop,daemon=True);thread.start()
    restores=[];error=None;started=time.monotonic();old=sys.argv
    try:
        if args.resource_observe_sampling=='on':restores.append(install_sampling_observer(events))
        if args.resource_fixture=='on':restores.append(install_fixture(fixture))
        import run_controlled
        sys.argv=[sys.argv[0]]+remaining
        return run_controlled.main()
    except BaseException as exc:error=type(exc).__name__;raise
    finally:
        sys.argv=old
        for restore in reversed(restores):restore()
        stop.set();thread.join(timeout=3.)
        result=dict(version='shared_GPU_fence_v1_20261028',cap_gib=args.resource_cap_gib,reserve_gib=args.resource_reserve_gib,
            total_bytes=total,fraction=fraction,resource_fixture=args.resource_fixture,wall_seconds=time.monotonic()-started,error_type=error,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            source_solver_and_samples_unchanged=True,only_own_torch_allocator_capped=True,
            CUDA_driver_and_other_process_memory_not_capped=True,
            sampling_synchronization_for_timing=args.resource_observe_sampling=='on')
        with out.with_name(out.name+'_resource_result.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)


if __name__=='__main__':main()
