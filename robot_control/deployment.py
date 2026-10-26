"""Offline policy/ControlModule boundary; no ROS publishers or motor enable calls.

The existing driver owns motor signs and parallel-ankle transmission. Inputs
here are named *joint-side* feedback. Root orientation is world-from-body wxyz,
angular velocity is body-frame rad/s. Motion tracking requires an independent,
time-aligned root-pose estimate. Reference poses must never fill missing sensors.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import numpy as np

from .robot_layout import ExplicitJointOrder, RobotLayout


def quaternion_matrix(q):
    q=np.asarray(q,dtype=np.float64)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.01:
        raise ValueError('Expected normalized finite quaternion wxyz')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def quat_multiply(a,b):
    w,x,y,z=a; v,i,j,k=b
    return np.array([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v])


def imu_to_body(world_from_imu_wxyz, angular_velocity_imu, body_from_imu_wxyz):
    """The mounting quaternion is required explicitly, never inferred from IMU data."""
    r=quaternion_matrix(body_from_imu_wxyz)
    quaternion_matrix(world_from_imu_wxyz)
    b=np.asarray(body_from_imu_wxyz,float); b=b/np.linalg.norm(b)
    w=np.asarray(angular_velocity_imu,float)
    if w.shape!=(3,) or not np.isfinite(w).all(): raise ValueError('Invalid IMU gyro')
    conjugate=b*np.array([1.,-1.,-1.,-1.])
    q=quat_multiply(np.asarray(world_from_imu_wxyz,float),conjugate)
    return q/np.linalg.norm(q),r@w


class PolicyBridge:
    """50Hz history and issued-action bookkeeping with strict named boundaries.

    A fault latches until reset. This class only builds a command object: the
    existing ControlModule's mode arbitration, watchdog and damping transition
    remain responsible for actually accepting or rejecting that command.
    """
    def __init__(self, control, hardware, *, interface='motion', max_age_s=.04):
        self.names=tuple(control['joint_names']); self.layout=RobotLayout(len(self.names))
        if self.layout.dof!=23 or interface not in ('motion','velocity'): raise ValueError('Expected Mojia 23DOF interface')
        self.interface=interface; self.dt=float(control['control_dt']);self.max_age=float(max_age_s)
        if not math.isclose(self.dt,.02) or self.max_age<=0: raise ValueError('Expected 50Hz and a positive age bound')
        self.nominal=self._vector(control['nominal'],'nominal')
        self.scale=self._vector(control['action_scale'],'action_scale',broadcast=True)
        self.kp=self._vector(control['stiffness'],'stiffness');self.kd=self._vector(control['damping'],'damping')
        self.lower=self._vector(control['lower'],'lower');self.upper=self._vector(control['upper'],'upper')
        self.speed=self._vector(control['velocity_limit'],'velocity_limit')
        self.effort=self._vector(control['effort_limit'],'effort_limit')
        if np.any(self.scale<=0) or np.any(self.lower>=self.upper) or np.any(self.speed<=0): raise ValueError('Invalid action/limit contract')
        if np.any(self.kp<0) or np.any(self.kd<0): raise ValueError('Negative control gain')
        entries={j['name']:j for j in hardware['joints']}
        if set(entries)!=set(self.names): raise ValueError('Incomplete hardware joint map')
        self.offset=np.array([entries[n]['control_offset_rad'] for n in self.names])
        self.parallel=np.array(['AnkleParallelTransmission' in entries[n]['transmission']['type'] for n in self.names])
        self.output_order=ExplicitJointOrder(self.names,hardware['control_joint_order'])
        self.reset()

    @classmethod
    def from_bundle(cls,bundle,hardware,*,interface='motion',allow_smoke=False):
        """Bind preprocessing/control constants to the exported learner contract.

        This creates an offline adapter, never permission to command hardware.
        Source binding still cannot certify real actuator saturation or timing.
        """
        if bundle.get('schema')!='mojia_policy_bundle_v1':raise ValueError('Wrong policy bundle schema')
        if bundle.get('smoke_test') and not allow_smoke:raise ValueError('Smoke bundle is verification-only')
        if interface not in bundle.get('trained_interfaces',[]):raise ValueError('Requested interface has no learned weights')
        control=bundle.get('control')
        if not control:raise ValueError('Policy bundle has no bound control parameters')
        digest=hashlib.sha256(json.dumps(control,sort_keys=True).encode()).hexdigest()
        if digest!=bundle['training_contract']['control_sha256']:raise ValueError('Control parameters do not match learned policy')
        if control['joint_names']!=bundle['training_contract']['joint_names']:raise ValueError('Policy joint order changed')
        return cls(control,hardware,interface=interface)

    def _vector(self,value,label,broadcast=False):
        a=np.asarray(value,dtype=np.float64)
        if broadcast and a.ndim==0:a=np.full(len(self.names),float(a))
        if a.shape!=(len(self.names),) or not np.isfinite(a).all():raise ValueError('Invalid '+label)
        return a.copy()

    def reset(self):
        self.history=deque(maxlen=self.layout.history);self.last_action=np.zeros(self.layout.dof)
        self.last_stamp=None;self.last_command_stamp=None
        self.last_target=None;self.state=None;self.fault=None;self.commanded=False

    def _fail(self,message):
        self.fault=message
        raise ValueError(message)

    def observe(self, *, names, position, velocity, quaternion_wxyz, omega_body,
                timestamp_s, now_s, feedback_at='driver_topic', root_position=None,
                root_velocity=None, pose_timestamp_s=None):
        if self.fault:self._fail('Latched fault; reset before resuming: '+self.fault)
        try:
            if not all(math.isfinite(x) for x in (timestamp_s,now_s)):raise ValueError('Nonfinite timestamp')
            age=now_s-timestamp_s
            if not 0<=age<=self.max_age:raise ValueError('Stale/future joint or IMU sample')
            if self.last_stamp is not None:
                dt=timestamp_s-self.last_stamp
                if not .5*self.dt<=dt<=1.5*self.dt:raise ValueError('History must advance once per 50Hz policy tick')
                if not self.commanded:raise ValueError('Previous observation has no accepted issued command')
            order=ExplicitJointOrder(names,self.names)
            q=self._vector(order.to_target(position),'position');dq=self._vector(order.to_target(velocity),'velocity')
            if feedback_at=='driver_topic':q=q-self.offset
            elif feedback_at!='control_module':raise ValueError('Unknown zero-offset boundary')
            quat=np.asarray(quaternion_wxyz,float);mat=quaternion_matrix(quat);quat=quat/np.linalg.norm(quat)
            w=np.asarray(omega_body,float)
            if w.shape!=(3,) or not np.isfinite(w).all():raise ValueError('Invalid body angular velocity')
            if self.interface=='motion':
                if root_position is None or root_velocity is None or pose_timestamp_s is None:
                    raise ValueError('Motion tracking requires a measured root pose/velocity and timestamp')
                if not math.isfinite(pose_timestamp_s) or abs(pose_timestamp_s-timestamp_s)>.5*self.dt:
                    raise ValueError('Root observer and joint/IMU sample are not synchronized')
            p=np.zeros(3) if root_position is None else np.asarray(root_position,float)
            v=np.zeros(3) if root_velocity is None else np.asarray(root_velocity,float)
            if p.shape!=(3,) or v.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(v).all():raise ValueError('Invalid root state')
            obs=np.concatenate((q-self.nominal,dq,w,-mat[2],self.last_action)).astype(np.float32)
            if not self.history:
                for _ in range(self.layout.history):self.history.append(obs.copy())
            else:self.history.append(obs)
            self.state=np.concatenate((p,quat,v,w,q,dq,self.last_action)).astype(np.float32)
            self.last_stamp=timestamp_s;self.commanded=False
            return np.stack(self.history)[None],self.state[None].copy()
        except (ValueError,TypeError) as e:self._fail(str(e))

    def make_command(self,action,*,now_s,output_at='control_module'):
        if self.fault:self._fail('Latched fault; reset before resuming: '+self.fault)
        try:
            if self.state is None or self.commanded:raise ValueError('Observe exactly once before issuing each action')
            if not math.isfinite(now_s) or not 0<=now_s-self.last_stamp<=self.max_age:raise ValueError('Action deadline missed')
            action=self._vector(action,'action');target=self.nominal+self.scale*action
            if np.any(target<self.lower) or np.any(target>self.upper):raise ValueError('Policy target exceeds joint limits')
            previous=self.state[13:36] if self.last_target is None else self.last_target
            if np.any(abs(target-previous)>self.speed*self.dt+1e-6):raise ValueError('Policy target exceeds configured slew limit')
            result=self._compose(target,self.state[13:36],self.state[36:59],now_s,output_at)
            self.last_action=action.copy();self.last_target=target;self.commanded=True
            # New observations alone must never renew an old policy command.
            # Use the observation that produced this accepted action, so inference
            # latency consumes the watchdog budget instead of extending it.
            self.last_command_stamp=self.last_stamp
            return result
        except (ValueError,TypeError) as e:self._fail(str(e))

    def _compose(self,target,q,dq,now_s,output_at):
        outgoing=target.copy()
        if output_at=='driver_topic':outgoing+=self.offset
        elif output_at!='control_module':raise ValueError('Unknown zero-offset boundary')
        # Parallel-ankle scalar motor gains are not joint-space gains. Compute
        # joint torques each servo tick; the driver applies its force Jacobian.
        tau=np.zeros(23);kp=self.kp.copy();kd=self.kd.copy()
        tau[self.parallel]=np.clip((self.kp*(target-q)-self.kd*dq)[self.parallel],
                                  -self.effort[self.parallel],self.effort[self.parallel])
        kp[self.parallel]=0.;kd[self.parallel]=0.
        reorder=lambda x:list(self.output_order.to_target(x))
        return dict(name=list(self.output_order.target_names),position=reorder(outgoing),
                    velocity=[0.]*23,effort=reorder(tau),stiffness=reorder(kp),damping=reorder(kd),
                    timestamp_s=now_s,boundary=output_at,servo_refresh_required_hz=1000,
                    driver_parallel_jacobian_validation_required=True,
                    hardware_actuation_equivalence_verified=False,
                    unresolved_actuation=['19 direct-MIT motors use firmware torque saturation; simulated limits unverified',
                                          'hardware servo 1000Hz versus simulated PD update200Hz; response must be validated'])

    def refresh_servo(self,*,names,position,velocity,timestamp_s,now_s,
                      feedback_at='driver_topic',output_at='control_module'):
        """1kHz target hold with fresh ankle PD, without advancing 50Hz history."""
        if self.fault:self._fail('Latched fault; reset before resuming: '+self.fault)
        try:
            if self.last_target is None:raise ValueError('No accepted policy target')
            if not all(math.isfinite(v) for v in (timestamp_s,now_s)):raise ValueError('Invalid servo timestamp')
            if not 0<=now_s-timestamp_s<=.005:raise ValueError('Servo feedback older than 5ms')
            if self.last_command_stamp is None or not 0<=now_s-self.last_command_stamp<=self.max_age:
                raise ValueError('Policy watchdog expired')
            order=ExplicitJointOrder(names,self.names)
            q=self._vector(order.to_target(position),'position');dq=self._vector(order.to_target(velocity),'velocity')
            if feedback_at=='driver_topic':q-=self.offset
            elif feedback_at!='control_module':raise ValueError('Unknown zero-offset boundary')
            return self._compose(self.last_target,q,dq,now_s,output_at)
        except (ValueError,TypeError) as e:self._fail(str(e))
