"""补充物理口径：车体坐标的有符号加速度和向量jerk，旧口径保持不变。"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import numpy as np

FEATURES=('velocity','lon_accel','lat_accel','jerk_norm')
EXPECTED_REFERENCE_SHA='eeb7be9486af31e12d8c2c321e41e16d7d48e8a7021d17b6378900cf2f383c6f'


@dataclass
class Previous:
    step:int
    position:np.ndarray
    velocity:np.ndarray|None=None
    acceleration:np.ndarray|None=None


class PhysicalAccumulator:
    def __init__(self):
        self.previous={};self.values={k:[] for k in FEATURES};self.invalid_states=0

    def observe(self,agent_id,step,state,dt):
        if not np.isfinite(dt) or dt<=0:raise ValueError('invalid_dt')
        state=np.asarray(state,float)
        if len(state)<5 or not np.isfinite(state[:5]).all():
            self.previous.pop(agent_id,None);self.invalid_states+=1;return
        position=state[:2];yaw=state[4];old=self.previous.get(agent_id)
        new=Previous(step,position.copy())
        if old is not None and step==old.step+1:
            velocity=(position-old.position)/dt;new.velocity=velocity
            self.values['velocity'].append(float(np.linalg.norm(velocity)))
            if old.velocity is not None:
                acceleration=(velocity-old.velocity)/dt;new.acceleration=acceleration
                forward=np.array([np.cos(yaw),np.sin(yaw)])
                left=np.array([-np.sin(yaw),np.cos(yaw)])
                self.values['lon_accel'].append(float(acceleration@forward))
                self.values['lat_accel'].append(float(acceleration@left))
                if old.acceleration is not None:
                    self.values['jerk_norm'].append(float(np.linalg.norm((acceleration-old.acceleration)/dt)))
        self.previous[agent_id]=new


def wasserstein_raw(left,right):
    """经验一维W1，积分两个经验CDF的差；不丢弃负数或尾部样本。"""
    a=np.sort(np.asarray(left,float));b=np.sort(np.asarray(right,float))
    a=a[np.isfinite(a)];b=b[np.isfinite(b)]
    if not len(a) or not len(b):return None
    points=np.sort(np.concatenate([a,b]));widths=np.diff(points)
    delta=np.abs(np.searchsorted(a,points[:-1],side='right')/len(a)-np.searchsorted(b,points[:-1],side='right')/len(b))
    return float(delta@widths)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def source_verified(expected,actual):
    if expected!=actual:raise ValueError('reference_source_payload_hash_mismatch')


def build(args):
    """固定读取同100个Waymo文件；数据集只读，所有派生文件写独立新目录。"""
    if args.output.exists():raise ValueError('preserve_existing_reference_run')
    if sha(args.legacy_reference)!=EXPECTED_REFERENCE_SHA:raise ValueError('legacy_reference_hash_mismatch')
    reference=json.loads(args.legacy_reference.read_text());sources=reference['sources']
    if len(sources)!=100 or reference['dt_seconds']!=.1:raise ValueError('frozen_reference_scope_mismatch')
    dataset_root=Path('/hqlab/dataset_nas1/waymo_perception_v_1_2_0').resolve()
    for item in sources:
        path=Path(item['path']).resolve()
        if dataset_root not in path.parents or not path.is_file():raise ValueError('source_not_readable_in_declared_dataset')
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='supplementary_body_frame_raw_W1_v1_20261027',reference_source_files=100,
        legacy_reference_sha256=EXPECTED_REFERENCE_SHA,legacy_source_manifest_sha256=reference['source_manifest_sha256'],
        source_sha256={str(p):sha(p) for p in [Path(__file__),args.legacy_builder,args.legacy_reference]},
        legacy_metric_unchanged=True,dt_seconds=.1,
        feature_definitions={'velocity':'norm of backward position-difference velocity',
            'lon_accel':'dot(backward difference velocity acceleration, current vehicle heading)',
            'lat_accel':'dot(backward difference velocity acceleration, current left normal)',
            'jerk_norm':'norm(backward difference global acceleration vector)'},
        distance='empirical raw W1 per feature, full signed support; no histogram tail exclusion',
        metric_aggregation='report four distances separately; no ad-hoc combined physical score',
        scene_aggregation='equal scene, pooled full, common observed prefix all reported',
        smoothing='none; finite differences sensitive to measurement noise, disclosed',
        filters='same TYPE_VEHICLE positive lidar points; contiguous frames only; no ego labels',
        CPU_only=True,new_API_requests=0,new_simulation_frames=0,started_at=time.time())
    save(args.output/'protocol.json',protocol)
    try:
        # 禁止TensorFlow占用GPU；不改环境、不安装依赖、不写数据集。
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.threading.set_intra_op_parallelism_threads(1)
        from waymo_open_dataset import dataset_pb2,label_pb2
        spec=importlib.util.spec_from_file_location('legacy_waymo_geometry',args.legacy_builder)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        accumulator=PhysicalAccumulator();receipts=[];total_frames=0;total_labels=0;gaps=0
        for item in sources:
            path=Path(item['path']);accumulator.previous.clear();digest=hashlib.sha256()
            origin_inverse=None;origin_yaw=None;last_timestamp=None;frames=0;labels=0
            options=tf.data.Options();options.threading.private_threadpool_size=1
            dataset=tf.data.TFRecordDataset(str(path),compression_type='').with_options(options)
            for step,raw in enumerate(dataset):
                payload=bytes(raw.numpy());digest.update(payload);frame=dataset_pb2.Frame();frame.ParseFromString(payload)
                transform=np.asarray(frame.pose.transform,float).reshape(4,4)
                if origin_inverse is None:origin_inverse=np.linalg.inv(transform);origin_yaw=float(np.arctan2(transform[1,0],transform[0,0]))
                stamp=int(frame.timestamp_micros)
                if last_timestamp is not None and not np.isclose((stamp-last_timestamp)/1e6,.1,rtol=0,atol=1e-3):
                    accumulator.previous.clear();gaps+=1
                last_timestamp=stamp
                for label in frame.laser_labels:
                    if int(label.type)!=int(label_pb2.Label.TYPE_VEHICLE) or int(label.num_lidar_points_in_box)<=0:continue
                    state=legacy._scene_state(frame,label,origin_inverse,origin_yaw)
                    if not np.isfinite(state).all():
                        accumulator.invalid_states+=1;accumulator.previous.pop(str(label.id),None);continue
                    accumulator.observe(str(label.id),step,state,.1);labels+=1
                frames+=1
            source_verified(item['record_payload_sha256'],digest.hexdigest())
            if frames!=item['frame_count'] or labels!=item['vehicle_label_count']:raise ValueError('reference_frame_or_label_count_mismatch')
            receipts.append(dict(path=str(path),payload_sha256=digest.hexdigest(),frames=frames,labels=labels))
            total_frames+=frames;total_labels+=labels
            save(args.output/'progress.json',dict(completed_files=len(receipts),planned_files=100,frames=total_frames,labels=total_labels,
                checked_at=time.time(),sample_counts={k:len(v) for k,v in accumulator.values.items()}))
        if total_frames!=reference['frame_count'] or total_labels!=reference['vehicle_label_count'] or gaps!=reference['discontinuous_interval_count']:
            raise ValueError('reference_total_scope_mismatch')
        arrays={k:np.asarray(v,float) for k,v in accumulator.values.items()}
        if any(not len(v) or not np.isfinite(v).all() for v in arrays.values()):raise ValueError('invalid_reference_samples')
        np.savez_compressed(args.output/'physical_samples.npz',**arrays)
        save(args.output/'reference_metadata.json',dict(protocol=protocol,source_receipts=receipts,
            completed_files=100,frames=total_frames,labels=total_labels,discontinuous_intervals=gaps,
            invalid_states=accumulator.invalid_states,sample_archive_sha256=sha(args.output/'physical_samples.npz'),
            feature_statistics={k:dict(samples=len(v),minimum=float(v.min()),maximum=float(v.max()),
                quantiles={str(q):float(np.quantile(v,q)) for q in [.01,.05,.5,.95,.99]},negative_fraction=float(np.mean(v<0))) for k,v in arrays.items()},
            limitations=['backward_finite_difference_noise','current_yaw_for_body_projection','reference_not_conditioned_on_synthetic_scene_distribution']))
        save(args.output/'stage_result.json',dict(status='completed',source_files_verified=100,frames=total_frames,labels=total_labels,
            reference_sha256=sha(args.output/'physical_samples.npz'),finished_at=time.time()))
    except Exception as error:
        save(args.output/'stage_result.json',dict(status='failed',error_type=type(error).__name__,finished_at=time.time()))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--legacy-reference',type=Path,required=True)
    parser.add_argument('--legacy-builder',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args())


if __name__=='__main__':main()
