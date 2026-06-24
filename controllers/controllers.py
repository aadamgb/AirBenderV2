import math
import torch

 
PTERM_SCALE = 0.032029
ITERM_SCALE = 0.244381
DTERM_SCALE = 0.000529

class BaseController:
    def map(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        pass

class SRT(BaseController):
    def __init__(self, max_thrust, **kwargs): 
        self.max_thrust = max_thrust
        
    def map(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return action * self.max_thrust
    
class lazyCTBR(BaseController):
    """
    Lazy implementation of PX4 rate controller
    Airmode: disabled  (If motor thurst is negative or higher than max we calamp)
    """ 
    def __init__(self,
                 mixer, 
                 max_thrust,
                 max_rates = (6.0, 6.0, 3.0), 
                 kp        = (45.0, 45.0, 45.0),
                 ki        = (8.0,   4.0,  8.0), 
                 kd        = (50.0, 65.0, 40.0),
                 device    = "cuda",
                  **kwargs):
        
        self.max_thrust = max_thrust
        self.max_rates = torch.tensor(max_rates, dtype=torch.float32, device=device)
        self.mixer_inv = torch.linalg.pinv(mixer)
        self.kp = torch.tensor(kp, dtype=torch.float32, device=device) * PTERM_SCALE
        self.ki = torch.tensor(ki, dtype=torch.float32, device=device) * ITERM_SCALE
        self.kd = torch.tensor(kd, dtype=torch.float32, device=device) * DTERM_SCALE

    def map(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        
        Fz_cmd = action[:, 0:1] * self.max_thrust * 4.0
        rates_cmd   = (2.0 * action[:, 1:4]  - 1.0) * self.max_rates
        error = rates_cmd - state[:, 10:13]
        tau_cmd = self.kp * error
        wrench = torch.cat([Fz_cmd, tau_cmd], dim=-1) 
        motor_thrusts = (wrench @ self.mixer_inv.T).clamp(0.0, self.max_thrust)    
                
        return motor_thrusts

class pidCTBR(BaseController):
    def __init__(self,
                 mixer,
                 max_thrust,
                 max_rates  = (6.0,  6.0,  3.0),
                #  max_rates  = (10.0,  10.0,  3.0),
                 kp         = (45.0, 45.0, 45.0),
                 ki         = (8.0,   4.0,  8.0),
                 kd         = (50.0, 65.0, 40.0),
                 int_limit  = (0.4,  0.4,  0.3),   
                 device     = "cuda",
                 **kwargs):

        self.max_thrust = max_thrust
        self.max_rates  = torch.tensor(max_rates, dtype=torch.float32, device=device)
        self.mixer_inv  = torch.linalg.pinv(mixer)
        self.kp = torch.tensor(kp, dtype=torch.float32, device=device) * PTERM_SCALE
        self.ki = torch.tensor(ki, dtype=torch.float32, device=device) * ITERM_SCALE
        self.kd = torch.tensor(kd, dtype=torch.float32, device=device) * DTERM_SCALE
        self.int_limit  = torch.tensor(int_limit, dtype=torch.float32, device=device)
        self.device     = device

        self.rate_int   = None  
        self.prev_rates = None  

    def reset(self, env_mask: torch.Tensor | None = None):
        if env_mask is None:
            if self.rate_int is not None:
                self.rate_int.zero_()
            self.prev_rates = None
        else:
            if self.rate_int is not None:
                self.rate_int[env_mask] = 0.0
            if self.prev_rates is not None:
                self.prev_rates[env_mask] = 0.0

    def map(self,
            state:  torch.Tensor,
            action: torch.Tensor,
            dt: float = 0.01,
            ) -> torch.Tensor:

        num_envs = state.shape[0]

        if self.rate_int is None:
            self.rate_int = torch.zeros(num_envs, 3, dtype=torch.float32, device=self.device)

        Fz_cmd    = action[:, 0:1] * self.max_thrust * 4
        rates_cmd = (2.0 * action[:, 1:4] - 1.0) * self.max_rates

        rates = state[:, 10:13]
        error = rates_cmd - rates

        p_term = self.kp * error

        if self.prev_rates is None:
            angular_accel = torch.zeros_like(rates)
        else:
            angular_accel = (rates - self.prev_rates) / dt
        self.prev_rates = rates.clone()

        d_term = -self.kd * angular_accel
                          
        self.rate_int += self.ki * error * dt 
        self.rate_int = self.rate_int.clamp(-self.int_limit, self.int_limit)

        tau_cmd = p_term + self.rate_int + d_term   
                     
        wrench        = torch.cat([Fz_cmd, tau_cmd], dim=-1)     
        motor_thrusts = (wrench @ self.mixer_inv.T).clamp(0.0, self.max_thrust)

        return motor_thrusts


class px4CTBR(BaseController):
    def __init__(self,
                 mixer,
                 max_thrust,
                 max_rates = (6.0, 6.0, 3.0),       
                 K   = (1.0,   1.0,   1.0),              # MC_xRATE_K
                 P   = (0.042,  0.042,   0.2),           # MC_xRATE_P
                 I   = (0.08,    0.08,   0.1),           # MC_xRATE_I
                 D   = (0.0015, 0.0015,  0.0),           # MC_xRATE_D
                 FF  = (0.0,      0.0,   0.0),           # MC_xRATE_FF
                 int_limit = (0.30, 0.30, 0.30),         # MC_xR_INT_LIM
                 device = "cuda",
                 **kwargs):

        self.max_thrust = max_thrust
        self.device     = device
        self.max_rates  = torch.tensor(max_rates, dtype=torch.float32, device=device)
        self.mixer_inv  = torch.linalg.pinv(mixer)

        K = torch.tensor(K, dtype=torch.float32, device=device)

        self.gain_p  = K * torch.tensor(P,  dtype=torch.float32, device=device)
        self.gain_i  = K * torch.tensor(I,  dtype=torch.float32, device=device)
        self.gain_d  = K * torch.tensor(D,  dtype=torch.float32, device=device)
        self.gain_ff =     torch.tensor(FF, dtype=torch.float32, device=device)
        self.lim_int =     torch.tensor(int_limit, dtype=torch.float32, device=device)

        self.i_factor_norm = math.radians(400.0)

        self.rate_int  = None   
        self._prev_rate = None   # only used if angular_accel is not given

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

    def map(self,
            state:  torch.Tensor,                        
            action: torch.Tensor,                        
            angular_accel: torch.Tensor | None = None,   
            dt: float = 0.01,
            ) -> torch.Tensor:

        N = state.shape[0]
        if self.rate_int is None:
            self.rate_int = torch.zeros(N, 3, dtype=torch.float32, device=self.device)

        Fz_cmd  = action[:, 0:1] * self.max_thrust * 4.0          
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

        wrench        = torch.cat([Fz_cmd, torque], dim=-1)
        motor_thrusts = (wrench @ self.mixer_inv.T).clamp(0.0, self.max_thrust)
        return motor_thrusts

    def _update_integral(self, rate_error, dt):
        i_factor = rate_error / self.i_factor_norm
        i_factor = torch.clamp(1.0 - i_factor * i_factor, min=0.0)

        rate_i = self.rate_int + i_factor * self.gain_i * rate_error * dt
        rate_i = rate_i.clamp(-self.lim_int, self.lim_int)

        new_int = torch.where(torch.isfinite(rate_i), rate_i, self.rate_int)

        self.rate_int = new_int


class px4CTBRv2(BaseController):
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

        # PX4 mixer scales, derived from the sign pattern of the physical mixer
        # rows [Fz, tau_x, tau_y, tau_z] so the convention always matches the
        # dynamics. For a symmetric X-quad these are exactly +/-1, which is what
        # PX4 uses. (Asymmetric frames would need normalized magnitudes, not signs.)
        mixer = mixer.to(device)
        self.thrust_scale = torch.sign(mixer[0])   # (4,)
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

    # ── rate controller (unchanged) ───────────────────────────────────────────
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
                  + self.gain_ff * rate_sp)            # normalized [-1,1]ish
        self._update_integral(rate_error, dt)

        # collective force -> normalized throttle (quadratic model)
        throttle = torch.sqrt(action[:, 0:1].clamp(min=0.0))   # (N,1)

        # PX4 airmode-disabled mixer, normalized [0,1] per motor
        motor_norm = self._mixer_px4(throttle, torque)         # (N,4)

        # normalized throttle -> physical per-motor force
        return motor_norm ** 2 * self.max_thrust

    def _update_integral(self, rate_error, dt):
        i_factor = rate_error / self.i_factor_norm
        i_factor = torch.clamp(1.0 - i_factor * i_factor, min=0.0)
        rate_i = self.rate_int + i_factor * self.gain_i * rate_error * dt
        rate_i = rate_i.clamp(-self.lim_int, self.lim_int)
        self.rate_int = torch.where(torch.isfinite(rate_i), rate_i, self.rate_int)

    # ── PX4 mix_airmode_disabled (vectorized) ─────────────────────────────────
    def _compute_desat_gain(self, outputs, desat, min_o, max_o, eps=1e-6):
        # outputs (N,4), desat (4,)  ->  gain (N,1)
        valid     = desat.abs() >= eps                       # (4,)
        desat_sfe = torch.where(valid, desat, torch.ones_like(desat))
        below = outputs < min_o                              # (N,4)
        above = outputs > max_o
        k_low  = (min_o - outputs) / desat_sfe               # (N,4)
        k_high = (max_o - outputs) / desat_sfe
        cand = torch.cat([k_low, k_high], dim=-1)            # (N,8)
        cond = torch.cat([below & valid, above & valid], dim=-1)
        k_min = torch.minimum(torch.zeros_like(outputs[:, :1]),
                              torch.where(cond, cand,  float("inf")).amin(-1, keepdim=True))
        k_max = torch.maximum(torch.zeros_like(outputs[:, :1]),
                              torch.where(cond, cand, -float("inf")).amax(-1, keepdim=True))
        return k_min + k_max                                 # (N,1)

    def _minimize_saturation(self, outputs, desat, min_o=0.0, max_o=1.0, reduce_only=False):
        k1 = self._compute_desat_gain(outputs, desat, min_o, max_o)
        if reduce_only:
            k1 = torch.where(k1 > 0.0, torch.zeros_like(k1), k1)   # skip if would increase
        outputs = outputs + k1 * desat
        k2 = 0.5 * self._compute_desat_gain(outputs, desat, min_o, max_o)
        if reduce_only:
            k2 = torch.where(k1 == 0.0, torch.zeros_like(k2), k2)  # honour the early-return
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
                   + thr   * self.thrust_scale)              # (N,4)

        outputs = self._minimize_saturation(outputs, self.thrust_scale, 0.0, 1.0, reduce_only=True)
        outputs = self._minimize_saturation(outputs, self.roll_scale,   0.0, 1.0, reduce_only=False)
        outputs = self._minimize_saturation(outputs, self.pitch_scale,  0.0, 1.0, reduce_only=False)
        outputs = self._mix_yaw(outputs, yaw)

        # the 2x-1 / 0.5x+0.5 round-trip collapses to a clamp
        return outputs.clamp(0.0, 1.0)

CONTROLLERS = {
    "srt":  SRT,
    "lazy_ctbr": lazyCTBR,
    "pid_ctbr": pidCTBR,
    "px4" : px4CTBR,
    "px4v2" : px4CTBRv2,
}