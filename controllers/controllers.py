import math
import torch
from utils.math import quat_to_rotmat, vee
class BaseController:
    def map(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        pass

    def detach_state(self) -> None:
        pass

class SRT(BaseController):
    def __init__(self, max_thrust, **kwargs): 
        self.max_thrust = max_thrust
        
    def map(self, state: torch.Tensor, action: torch.Tensor,  **kwargs) -> torch.Tensor:
        return action * self.max_thrust
    
class px4CTBR(BaseController):
    def __init__(self,
                 mixer,
                 max_thrust,
                 max_rates = (6.0, 6.0, 3.0),
                 K   = (1.0,    1.0,    1.0),
                 P   = (0.042,  0.042,  0.2),
                 I   = (0.08,   0.08,   0.1),
                 D   = (0.0015, 0.0015, 0.0),
                 FF  = (0.0,    0.0,    0.0),
                 int_limit = (0.30, 0.30, 0.30),
                 device = "cuda",
                 **kwargs):

        self.max_thrust = max_thrust
        self.device     = device
        self.max_rates  = torch.tensor(max_rates, dtype=torch.float32, device=device)

        K = torch.tensor(K, dtype=torch.float32, device=device)
        self.gain_p  = K * torch.tensor(P,  dtype=torch.float32, device=device)
        self.gain_i  = K * torch.tensor(I,  dtype=torch.float32, device=device)
        self.gain_d  = K * torch.tensor(D,  dtype=torch.float32, device=device)
        self.gain_ff =     torch.tensor(FF, dtype=torch.float32, device=device)
        self.lim_int =     torch.tensor(int_limit, dtype=torch.float32, device=device)
        self.i_factor_norm = math.radians(400.0)

        mixer = mixer.to(device)
        self.thrust_scale = torch.sign(mixer[0])   
        self.roll_scale   = torch.sign(mixer[1])
        self.pitch_scale  = torch.sign(mixer[2])
        self.yaw_scale    = torch.sign(mixer[3])

        self.rate_int   = None
        self._prev_rate = None

    def reset(self, env_mask: torch.Tensor | None = None):
        if env_mask is None:
            if self.rate_int is not None:
                self.rate_int.zero_()
            self._prev_rate = None
        else:
            if self.rate_int is not None:
                self.rate_int[env_mask] = 0.0
            if self._prev_rate is not None:
                self._prev_rate[env_mask] = 0.0

    def detach_state(self) -> None:
        if self.rate_int is not None:
            self.rate_int = self.rate_int.detach()
        if self._prev_rate is not None:
            self._prev_rate = self._prev_rate.detach()

    def map(self, state, action, angular_accel=None, dt: float = 0.01):
        N = state.shape[0]
        if self.rate_int is None:
            self.rate_int = torch.zeros(N, 3, dtype=torch.float32, device=self.device)

        rate_sp = (2.0 * action[:, 1:4] - 1.0) * self.max_rates
        rate    = state[:, 10:13]

        if angular_accel is None:
            if self._prev_rate is None:
                angular_accel = torch.zeros_like(rate)
            else:
                angular_accel = (rate - self._prev_rate) / dt
            self._prev_rate = rate.clone()

        rate_error = rate_sp - rate
        torque = (self.gain_p  * rate_error
                  + self.rate_int
                  - self.gain_d  * angular_accel
                  + self.gain_ff * rate_sp)            
        self._update_integral(rate_error, dt)

        throttle = action[:, 0:1].clamp(min=0.0)   
        motor_norm = self._mixer_px4(throttle, torque)         

        return motor_norm * self.max_thrust

    def _update_integral(self, rate_error, dt):
        i_factor = rate_error / self.i_factor_norm
        i_factor = torch.clamp(1.0 - i_factor * i_factor, min=0.0)
        rate_i = self.rate_int + i_factor * self.gain_i * rate_error * dt
        rate_i = rate_i.clamp(-self.lim_int, self.lim_int)
        self.rate_int = torch.where(torch.isfinite(rate_i), rate_i, self.rate_int)

    def _compute_desat_gain(self, outputs, desat, min_o, max_o, eps=1e-6):
        valid     = desat.abs() >= eps                       
        desat_sfe = torch.where(valid, desat, torch.ones_like(desat))
        below = outputs < min_o                              
        above = outputs > max_o
        k_low  = (min_o - outputs) / desat_sfe               
        k_high = (max_o - outputs) / desat_sfe
        cand = torch.cat([k_low, k_high], dim=-1)            
        cond = torch.cat([below & valid, above & valid], dim=-1)
        k_min = torch.minimum(torch.zeros_like(outputs[:, :1]),
                              torch.where(cond, cand,  float("inf")).amin(-1, keepdim=True))
        k_max = torch.maximum(torch.zeros_like(outputs[:, :1]),
                              torch.where(cond, cand, -float("inf")).amax(-1, keepdim=True))
        return k_min + k_max                                 

    def _minimize_saturation(self, outputs, desat, min_o=0.0, max_o=1.0, reduce_only=False):
        k1 = self._compute_desat_gain(outputs, desat, min_o, max_o)
        if reduce_only:
            k1 = torch.where(k1 > 0.0, torch.zeros_like(k1), k1)  
        outputs = outputs + k1 * desat
        k2 = 0.5 * self._compute_desat_gain(outputs, desat, min_o, max_o)
        if reduce_only:
            k2 = torch.where(k1 == 0.0, torch.zeros_like(k2), k2)  
        return outputs + k2 * desat

    def _mix_yaw(self, outputs, yaw):
        outputs = outputs + yaw * self.yaw_scale
        outputs = self._minimize_saturation(outputs, self.yaw_scale, 0.0, 1.15, reduce_only=False)
        outputs = self._minimize_saturation(outputs, self.thrust_scale, 0.0, 1.0, reduce_only=True)
        return outputs

    def _mixer_px4(self, throttle, torque):
        roll  = torque[:, 0:1].clamp(-1.0, 1.0)
        pitch = torque[:, 1:2].clamp(-1.0, 1.0)
        yaw   = torque[:, 2:3].clamp(-1.0, 1.0)
        thr   = throttle.clamp(0.0, 1.0)

        # mix without yaw
        outputs = (roll  * self.roll_scale
                   + pitch * self.pitch_scale
                   + thr   * self.thrust_scale)    

        # Airmode Disabled
        outputs = self._minimize_saturation(outputs, self.thrust_scale, 0.0, 1.0, reduce_only=True)
        outputs = self._minimize_saturation(outputs, self.roll_scale,   0.0, 1.0, reduce_only=False)
        outputs = self._minimize_saturation(outputs, self.pitch_scale,  0.0, 1.0, reduce_only=False)
        outputs = self._mix_yaw(outputs, yaw)

        return outputs.clamp(0.0, 1.0)

class so3LVHR_CTBR(BaseController):
    def __init__(self,
                 mixer,
                 max_thrust,
                 mass,
                 gravity,
                 max_rates   = (6.0, 6.0, 3.0),
                 v_max       = (10.0, 10.0, 3.0),   
                 yawrate_max = 3.0,                
                 k_v         = (3.0, 3.0, 4.0),    
                 k_R         = (8.0, 8.0, 1.0),    
                 inner       = None,
                 device      = "cuda",
                 **kwargs):
        self.device     = device
        self.mass       = mass
        self.g          = gravity
        self.max_thrust = max_thrust
        self.max_rates  = torch.tensor(max_rates, dtype=torch.float32, device=device)
        self.v_max      = torch.tensor(v_max,     dtype=torch.float32, device=device)
        self.yawrate_max = yawrate_max
        self.k_v        = torch.tensor(k_v, dtype=torch.float32, device=device)
        self.k_R        = torch.tensor(k_R, dtype=torch.float32, device=device)
        self.inner      = inner if inner is not None else \
                          px4CTBR(mixer, max_thrust, max_rates=max_rates, device=device)

    def reset(self, env_mask=None):
        self.inner.reset(env_mask)

    def detach_state(self) -> None:
        self.inner.detach_state()

    def map(self, state, action, dt: float = 0.01):
        N = state.shape[0]
        v = state[:, 3:6]                       
        R = quat_to_rotmat(state[:, 6:10])      

        v_cmd    = (2.0 * action[:, 0:3] - 1.0) * self.v_max         
        yaw_rate = (2.0 * action[:, 3:4] - 1.0) * self.yawrate_max    

        e3 = torch.zeros(N, 3, device=self.device); e3[:, 2] = 1.0

        e_v     = v_cmd - v
        acc_des = self.k_v * e_v + self.g * e3                         

        b3_des = acc_des / acc_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b3     = R[:, :, 2]
        f_coll = (self.mass * (acc_des * b3).sum(-1, keepdim=True)).clamp_min(0.0)

        yaw  = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        b1_c = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
        b2_des = torch.cross(b3_des, b1_c, dim=-1)
        b2_des = b2_des / b2_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b1_des = torch.cross(b2_des, b3_des, dim=-1)
        R_des  = torch.stack([b1_des, b2_des, b3_des], dim=-1)      
  
        e_R = vee(0.5 * (R_des.transpose(1, 2) @ R - R.transpose(1, 2) @ R_des))

        w_des_world = torch.cat([torch.zeros(N, 2, device=self.device), yaw_rate], dim=-1)
        w_ff = (R.transpose(1, 2) @ R_des @ w_des_world.unsqueeze(-1)).squeeze(-1)

        omega_cmd = -self.k_R * e_R + w_ff  

        a0      = (f_coll / (4.0 * self.max_thrust)).clamp(0.0, 1.0)
        a_rates = ((omega_cmd / self.max_rates + 1.0) * 0.5).clamp(0.0, 1.0)
        ctbr_action = torch.cat([a0, a_rates], dim=-1)                

        return self.inner.map(state, ctbr_action, dt=dt)

class so3LVHRg_CTBR(so3LVHR_CTBR):

    def __init__(self,
                 mixer,
                 max_thrust,
                 mass,
                 gravity,
                 max_rates   = (6.0, 6.0, 3.0),
                 v_max       = (10.0, 10.0, 3.0),
                 yawrate_max = 3.0,
                 k_v_min     = (0.5, 0.5, 0.5),
                 k_v_max     = (6.0, 6.0, 8.0),
                 k_R_min     = (1.0, 1.0, 0.2),
                 k_R_max     = (16.0, 16.0, 2.0),
                 inner       = None,
                 device      = "cuda",
                 **kwargs):
        super().__init__(mixer, max_thrust, mass, gravity,
                         max_rates=max_rates, v_max=v_max, yawrate_max=yawrate_max,
                         inner=inner, device=device, **kwargs)
        self.k_v_min = torch.tensor(k_v_min, dtype=torch.float32, device=device)
        self.k_v_max = torch.tensor(k_v_max, dtype=torch.float32, device=device)
        self.k_R_min = torch.tensor(k_R_min, dtype=torch.float32, device=device)
        self.k_R_max = torch.tensor(k_R_max, dtype=torch.float32, device=device)

    def map(self, state, action, dt: float = 0.01):
        N = state.shape[0]
        v = state[:, 3:6]
        R = quat_to_rotmat(state[:, 6:10])

        v_cmd    = (2.0 * action[:, 0:3] - 1.0) * self.v_max
        yaw_rate = (2.0 * action[:, 3:4] - 1.0) * self.yawrate_max

        k_v = self.k_v_min + action[:, 4:7]  * (self.k_v_max - self.k_v_min)
        k_R = self.k_R_min + action[:, 7:10] * (self.k_R_max - self.k_R_min)

        e3 = torch.zeros(N, 3, device=self.device); e3[:, 2] = 1.0

        e_v     = v_cmd - v
        acc_des = k_v * e_v + self.g * e3

        b3_des = acc_des / acc_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b3     = R[:, :, 2]
        f_coll = (self.mass * (acc_des * b3).sum(-1, keepdim=True)).clamp_min(0.0)

        yaw  = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        b1_c = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
        b2_des = torch.cross(b3_des, b1_c, dim=-1)
        b2_des = b2_des / b2_des.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b1_des = torch.cross(b2_des, b3_des, dim=-1)
        R_des  = torch.stack([b1_des, b2_des, b3_des], dim=-1)

        e_R = vee(0.5 * (R_des.transpose(1, 2) @ R - R.transpose(1, 2) @ R_des))

        w_des_world = torch.cat([torch.zeros(N, 2, device=self.device), yaw_rate], dim=-1)
        w_ff = (R.transpose(1, 2) @ R_des @ w_des_world.unsqueeze(-1)).squeeze(-1)

        omega_cmd = -k_R * e_R + w_ff

        a0      = (f_coll / (4.0 * self.max_thrust)).clamp(0.0, 1.0)
        a_rates = ((omega_cmd / self.max_rates + 1.0) * 0.5).clamp(0.0, 1.0)
        ctbr_action = torch.cat([a0, a_rates], dim=-1)

        return self.inner.map(state, ctbr_action, dt=dt)

CONTROLLERS = {
    "srt":  SRT,
    "px4" : px4CTBR,
    "so3" : so3LVHR_CTBR,
    "so3+g" : so3LVHRg_CTBR,
}