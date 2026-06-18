import torch
import numpy as np

# ── PyTorch (batched, primary) ────────────────────────────────────────────────

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """q: (..., 4) [w,x,y,z] → R: (..., 3, 3)"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        torch.stack([1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)], -1),
        torch.stack([  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)], -1),
        torch.stack([  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)], -1),
    ], -2)


def quat_derivative(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """q: (..., 4) [w,x,y,z], w: (..., 3) → q_dot: (..., 4)"""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    return 0.5 * torch.stack([
        -qx*wx - qy*wy - qz*wz,
         qw*wx + qy*wz - qz*wy,
         qw*wy - qx*wz + qz*wx,
         qw*wz + qx*wy - qy*wx,
    ], -1)


def rpy_to_rotmat(rpy_deg: torch.Tensor) -> torch.Tensor:
    """rpy_deg: (..., 3) degrees → R: (..., 3, 3).  Convention: Rz @ Ry @ Rx"""
    rpy = torch.deg2rad(rpy_deg)
    r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    z, o = torch.zeros_like(r), torch.ones_like(r)
    Rx = torch.stack([torch.stack([o,  z,   z], -1), torch.stack([z,  cr, -sr], -1), torch.stack([z, sr, cr], -1)], -2)
    Ry = torch.stack([torch.stack([cp, z,  sp], -1), torch.stack([z,   o,   z], -1), torch.stack([-sp, z, cp], -1)], -2)
    Rz = torch.stack([torch.stack([cy, -sy, z], -1), torch.stack([sy, cy,   z], -1), torch.stack([z,  z,  o], -1)], -2)
    return Rz @ Ry @ Rx


# ── NumPy (single-sample, used by QuadrotorEnv and renderer) ─────────────────

def quat_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    """q: [w, x, y, z] → R: (3, 3)"""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def rpy_to_rotmat_np(rpy_deg: np.ndarray) -> np.ndarray:
    """rpy_deg: (3,) degrees → R: (3, 3).  Convention: Rz @ Ry @ Rx"""
    r, p, y = np.radians(rpy_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx
