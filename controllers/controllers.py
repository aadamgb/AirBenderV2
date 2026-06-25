import math
import torch
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
        throttle = torch.sqrt(action[:, 0:1].clamp(min=0.0))   

        # PX4 airmode-disabled mixer, normalized [0,1] per motor
        motor_norm = self._mixer_px4(throttle, torque)         

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
                   + thr   * self.thrust_scale)             

        outputs = self._minimize_saturation(outputs, self.thrust_scale, 0.0, 1.0, reduce_only=True)
        outputs = self._minimize_saturation(outputs, self.roll_scale,   0.0, 1.0, reduce_only=False)
        outputs = self._minimize_saturation(outputs, self.pitch_scale,  0.0, 1.0, reduce_only=False)
        outputs = self._mix_yaw(outputs, yaw)

        # the 2x-1 / 0.5x+0.5 round-trip collapses to a clamp
        return outputs.clamp(0.0, 1.0)

CONTROLLERS = {
    "srt":  SRT,
    "px4" : px4CTBR,
}