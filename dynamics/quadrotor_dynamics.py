import math
import torch
from utils.math import quat_to_rotmat, quat_derivative


def _deriv(p, v, q, w, Fz, tau, m, J, G):
    # p,v: (N,3)  q: (N,4)  w: (N,3)  Fz: (N,)  tau: (N,3)
    R = quat_to_rotmat(q)                              # (N, 3, 3)
    thrust_world = Fz.unsqueeze(-1) * R[..., 2]       # z-column of R scaled by Fz → (N, 3)
    p_dot = v
    v_dot = thrust_world / m + G
    w_dot = (tau - torch.linalg.cross(w, w * J)) / J  # Euler: (τ - ω × Jω) / J
    q_dot = quat_derivative(q, w)
    return p_dot, v_dot, q_dot, w_dot


def integrate_rk4(dt, p, v, q, w, Fz, tau, m, J, G):
    h2 = dt * 0.5
    pd1, vd1, qd1, wd1 = _deriv(p,         v,         q,         w,         Fz, tau, m, J, G)
    pd2, vd2, qd2, wd2 = _deriv(p+h2*pd1,  v+h2*vd1,  q+h2*qd1,  w+h2*wd1,  Fz, tau, m, J, G)
    pd3, vd3, qd3, wd3 = _deriv(p+h2*pd2,  v+h2*vd2,  q+h2*qd2,  w+h2*wd2,  Fz, tau, m, J, G)
    pd4, vd4, qd4, wd4 = _deriv(p+dt*pd3,  v+dt*vd3,  q+dt*qd3,  w+dt*wd3,  Fz, tau, m, J, G)
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

    def __init__(self, mass, inertia, length, gravity, dt, max_thrust, torque_const, device='cuda'):
        self.dt = dt
        self.mass = mass
        self.max_thrust = max_thrust
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')

        s = math.sin(math.radians(45.0))
        c = math.cos(math.radians(45.0))
        l, ct = length, torque_const

        # Maps per-motor thrust [u1..u4] → [Fz, τx, τy, τz]
        self.mixer = torch.tensor([
            [ 1.0,   1.0,  1.0,  1.0],
            [-l*s,   l*s,  l*s, -l*s],
            [-l*c,   l*c, -l*c,  l*c],
            [-ct,   -ct,   ct,   ct ],
        ], dtype=torch.float32, device=self.device)  # (4, 4)

        self.J = torch.tensor(inertia,         dtype=torch.float32, device=self.device)
        self.G = torch.tensor([0., 0., -gravity], dtype=torch.float32, device=self.device)

    # @torch.no_grad()
    def propagate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state:  (N, 13)
        action: (N,  4) ∈ [0, 1]
        → next state: (N, 13)
        """
        p, v, q, w = state[:, 0:3], state[:, 3:6], state[:, 6:10], state[:, 10:13]
        wrench = (action * self.max_thrust) @ self.mixer.T  # (N, 4)
        Fz, tau = wrench[:, 0], wrench[:, 1:4]
        p_n, v_n, q_n, w_n = integrate_rk4(self.dt, p, v, q, w, Fz, tau, self.mass, self.J, self.G)
        return torch.cat([p_n, v_n, q_n, w_n], dim=-1)
