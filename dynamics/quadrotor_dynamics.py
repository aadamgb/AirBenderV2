import math
import torch
from utils.math import quat_to_rotmat, quat_derivative

def _deriv(p, v, q, w, f_b, tau, m, J, G):
    R = quat_to_rotmat(q)                                 
    f_world = (R @ f_b.unsqueeze(-1)).squeeze(-1)
    p_dot = v
    v_dot = f_world / m + G
    w_dot = (tau - torch.linalg.cross(w, w * J)) / J 
    q_dot = quat_derivative(q, w)
    return p_dot, v_dot, q_dot, w_dot


def integrate_rk4(dt, p, v, q, w, f_b, tau, m, J, G):
    h2 = dt * 0.5
    pd1, vd1, qd1, wd1 = _deriv(p,         v,         q,         w,         f_b, tau, m, J, G)
    pd2, vd2, qd2, wd2 = _deriv(p+h2*pd1,  v+h2*vd1,  q+h2*qd1,  w+h2*wd1,  f_b, tau, m, J, G)
    pd3, vd3, qd3, wd3 = _deriv(p+h2*pd2,  v+h2*vd2,  q+h2*qd2,  w+h2*wd2,  f_b, tau, m, J, G)
    pd4, vd4, qd4, wd4 = _deriv(p+dt*pd3,  v+dt*vd3,  q+dt*qd3,  w+dt*wd3,  f_b, tau, m, J, G)
    c = dt / 6.0
    p_n = p + c * (pd1 + 2*pd2 + 2*pd3 + pd4)
    v_n = v + c * (vd1 + 2*vd2 + 2*vd3 + vd4)
    w_n = w + c * (wd1 + 2*wd2 + 2*wd3 + wd4)
    q_n = q + c * (qd1 + 2*qd2 + 2*qd3 + qd4)
    q_n = q_n / q_n.norm(dim=-1, keepdim=True)
    return p_n, v_n, q_n, w_n

class QuadrotorDynamics:
    """
    Batched rigid-body quadrotor dynamics on a chosen torch device.

    State  : (N, 13)  [p(3) | v(3) | q(4) | ω(3)]
    Action : (N,  4)  per-motor thrust fraction ∈ [0, 1]
    """

    def __init__(self, mass, inertia, length, angle, torque_const, tau_m, eta, Cd, rho, gravity, dt, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
        self.G = torch.tensor([0., 0., -gravity], dtype=torch.float32, device=self.device)
        self.dt = dt
        
        self.set_params(mass, inertia, length, angle, torque_const, tau_m, eta, Cd, rho)

    def propagate(self, state: torch.Tensor, thrust_cmd: torch.Tensor) -> torch.Tensor:
        """
        state:  (N, 17)
        Rotor Thrusts: (N,  4) 
        """
        # Unpack state
        p   = state[:, 0:3]
        v   = state[:, 3:6]
        q   = state[:, 6:10]
        w   = state[:, 10:13]
        Omega = state[:, 13:17]

        # Simulate motor delay
        Omega_n, thrust_n = self._motor_dynamics(Omega, thrust_cmd)

        # Get body forces                          
        wrench = torch.matmul(self.mixer, thrust_n.unsqueeze(-1)).squeeze(-1)  
        f_z, tau = wrench[:, 0], wrench[:, 1:4]

        f_drag = self._compute_drag(v, q)
        f_thrust = torch.zeros_like(f_drag)
        f_thrust[:, 2] = f_z

        f_body = f_drag + f_thrust

        # Integrate 
        p_n, v_n, q_n, w_n = integrate_rk4(self.dt, p, v, q, w, f_body, tau, self.mass, self.J, self.G)
        return torch.cat([p_n, v_n, q_n, w_n, Omega_n], dim=-1)
    
    def _compute_drag(self, v, q):
        R = quat_to_rotmat(q)
        v_body = (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
        f_drag = - 0.5 * self.rho * v_body.abs() * v_body * self.Cd # NOTE: I think Cd is already Cd*area but check with Rob
        return f_drag
    
    def _motor_dynamics(self, Omega, thrust_cmd):
        # a0 = 4.5e-8                                     #TODO: Remove hardcode...
        a0 = 4e-6                                         #NOTE: a0 is actually arbitrary here
        Omega_cmd =  torch.sqrt((thrust_cmd / a0).clamp(min=1e-3))
        Omega_dot = (Omega_cmd - Omega) / self.tau_m
        Omega_n = Omega + Omega_dot * self.dt
        thrust_n = a0 * Omega_n **2
        return Omega_n, thrust_n
    
    @staticmethod
    def _build_mixer(eta: torch.Tensor, length: torch.Tensor, angle: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        """Maps per-motor thrust [u1..u4] → [Fz, τx, τy, τz], built from per-rotor arm length(s)."""
        sign_x = torch.tensor([-1.,  1.,  1., -1.], device=length.device)
        sign_y = torch.tensor([-1.,  1., -1.,  1.], device=length.device)
        sign_z = torch.tensor([-1., -1.,  1.,  1.], device=length.device)
 
        angle_rad = torch.deg2rad(angle)
        s, c = torch.sin(angle_rad), torch.cos(angle_rad)

        row_Fz  = eta
        row_tx  = length * s * sign_x
        row_ty  = length * c * sign_y
        row_tz  = (ct * sign_z) * torch.ones_like(length)
        return torch.stack([row_Fz, row_tx, row_ty, row_tz], dim=-2)  
    
    def set_params(self,
                   mass, inertia, 
                   length, angle, 
                   torque_const, 
                   tau_m, eta,
                   Cd, rho,
                   idx: torch.Tensor = None) -> None:

        mixer = self._build_mixer(eta, length, angle, torque_const) 
    
        if idx is None:
            self.mass              = mass
            self.J                 = inertia
            self.length            = length
            self.angle             = angle
            self.torque_const      = torque_const
            self.tau_m             = tau_m
            self.eta               = eta
            self.Cd                = Cd
            self.rho               = rho
            self.mixer             = mixer
        else:
            self.mass[idx]         = mass
            self.J[idx]            = inertia
            self.length[idx]       = length
            self.angle[idx]        = angle
            self.torque_const[idx] = torque_const
            self.tau_m[idx]        = tau_m
            self.eta[idx]          = eta
            self.Cd[idx]           = Cd
            self.rho[idx]           = rho
            self.mixer[idx]         = mixer

    def print_params(self) -> None:
        """Prints current physical parameters: mass, inertia (J), arm length, torque constant, and angle."""
        def to_cpu(x):
                return x.detach().cpu().numpy()
        
        print("Mass:\n", to_cpu(self.mass))
        print("\n Inertia (J):\n", to_cpu(self.J))
        print("\n Length:\n", to_cpu(self.length))
        print("\n Angle (deg):\n", to_cpu(self.angle))
        print("\n Torque_const:\n", to_cpu(self.torque_const))
        print("\n Tau motor:\n", to_cpu(self.tau_m))
        print("\n Eta:\n", to_cpu(self.eta))
        print("\n Cd:\n", to_cpu(self.Cd))
        print("\n rho:\n", to_cpu(self.rho))
