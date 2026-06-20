from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import csv

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from dynamics.quadrotor_dynamics import QuadrotorDynamics
from misc.loader import load_gates_from_yaml
from utils.math import rpy_to_rotmat, rpy_to_rotmat_np

# ── Shared physics / track constants ─────────────────────────────────────────
GRAVITY   = 9.81
MASS      = 1.21
INERTIA   = (0.007, 0.007, 0.013)
LENGTH    = 0.15
MAX_T     = 20.0
DT        = 0.01
TORQUE_C  = 0.012

X_THRESH        = 3.0
Y_THRESH        = 8.0
Z_THRESH        = 3.0
GATE_HALF_W     = 0.6
GATE_HALF_H     = 0.6
MAX_STEPS       = 1200

TRACK_PATH = "misc/racing_tracks/fig8.yaml"
PPO_LOG_DIR = "/home/adame/AirBender/outputs/ppo_logs"


# ─────────────────────────────────────────────────────────────────────────────
# GPU-batched vectorised environment  (training)
# ─────────────────────────────────────────────────────────────────────────────

class QuadrotorVecEnv(VecEnv):
    """
    Fully GPU-batched SB3 VecEnv.  All N environments run in parallel as
    batched tensor ops on `device`.  No subprocess overhead.

    State per env : (13,)  [p(3) | v(3) | q(4) | ω(3)]
    Action        : (4,)   per-motor thrust fraction ∈ [0, 1]
    Observation   : (17,)  see _get_obs
    """

    def __init__(self, num_envs: int = 256, device: str = "cuda", track_path: str = TRACK_PATH):
        obs_high = np.array([
            X_THRESH*2, Y_THRESH*2, Z_THRESH*2,           # rel_pos
            40., 40., 40.,                                 # rel_vel
            1., 1., 1., 1.,                               # q
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,                     # ω
            1., 1., 1.,                                   # gate_normal
            float(64),                                    # gate_idx
        ], dtype=np.float32)

        observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        action_space      = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        self.render_mode  = None
        super().__init__(num_envs, observation_space, action_space)

        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        # Gate data (precomputed, static)
        gates_pos_np, gates_rpy_np = load_gates_from_yaml(track_path)
        self.n_gates  = len(gates_pos_np)
        gates_pos_t   = torch.tensor(gates_pos_np, dtype=torch.float32, device=self.device)
        gates_rpy_t   = torch.tensor(gates_rpy_np, dtype=torch.float32, device=self.device)
        gates_R_t     = rpy_to_rotmat(gates_rpy_t)          # (G, 3, 3)

        self.register_buffer = lambda name, t: setattr(self, name, t)
        self.gates_pos    = gates_pos_t                     # (G, 3)
        self.gates_R      = gates_R_t                       # (G, 3, 3)
        self.gates_normal = gates_R_t[:, :, 1]             # (G, 3)  y-axis forward

        self.dynamics = QuadrotorDynamics(
            mass=MASS, inertia=INERTIA, length=LENGTH, gravity=GRAVITY,
            dt=DT, max_thrust=MAX_T, torque_const=TORQUE_C,
            device=str(self.device),
        )

        # Per-env tensors
        N = num_envs
        self.states     = torch.zeros(N, 13, dtype=torch.float32, device=self.device)
        self.gate_idx   = torch.zeros(N, dtype=torch.long,        device=self.device)
        self.prev_pos   = torch.zeros(N, 3,  dtype=torch.float32, device=self.device)
        self.step_count = torch.zeros(N, dtype=torch.long,        device=self.device)

        self._pending_actions: Optional[torch.Tensor] = None

        self._reset_envs(torch.ones(N, dtype=torch.bool, device=self.device))

        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def _reset_envs(self, mask: torch.Tensor) -> None:
        """Reset the environments where mask[i] is True."""
        n = int(mask.sum().item())
        if n == 0:
            return
        idx = mask.nonzero(as_tuple=True)[0]

        gate_ids = torch.randint(0, self.n_gates, (n,), device=self.device)
        self.gate_idx[idx] = gate_ids

        gate_R   = self.gates_R[gate_ids]    # (n, 3, 3)
        gate_pos = self.gates_pos[gate_ids]  # (n, 3)

        offset = torch.tensor([0., -1., 0.], dtype=torch.float32, device=self.device).expand(n, -1)
        noise  = torch.randn(n, 3, dtype=torch.float32, device=self.device) * 0.2
        # start  = gate_pos + (gate_R @ (offset + noise).unsqueeze(-1)).squeeze(-1)
        # start  = gate_pos + (gate_R @ (offset + noise).unsqueeze(-1)).squeeze(-1)

        self.states[idx]       = 0.0
        self.states[idx, 0:3]  = 0.0
        self.states[idx, 3:6]  = (torch.rand(n, 3, device=self.device) - 0.5)  # vel ∈ [-0.5, 0.5]
        self.states[idx, 6]    = 1.0   # qw = 1 (identity rotation)
        self.prev_pos[idx]     = 0.0
        self.step_count[idx]   = 0

    # ── Observation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _get_obs(self) -> np.ndarray:
        p, v, q, w = self.states[:, 0:3], self.states[:, 3:6], self.states[:, 6:10], self.states[:, 10:13]
        gate_R   = self.gates_R[self.gate_idx]      # (N, 3, 3)
        gate_pos = self.gates_pos[self.gate_idx]    # (N, 3)
        RT       = gate_R.transpose(-1, -2)         # (N, 3, 3)

        rel_pos    = (RT @ (p - gate_pos).unsqueeze(-1)).squeeze(-1)  # (N, 3)
        rel_vel    = (RT @ v.unsqueeze(-1)).squeeze(-1)                # (N, 3)
        gate_norm  = self.gates_normal[self.gate_idx]                  # (N, 3)
        gate_idx_f = self.gate_idx.float().unsqueeze(-1)               # (N, 1)

        obs = torch.cat([rel_pos, rel_vel, q, w, gate_norm, gate_idx_f], dim=-1)
        return obs.cpu().numpy().astype(np.float32)

    # ── VecEnv interface ──────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self._get_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def step_wait(self):
        prev_pos = self.states[:, 0:3].clone()

        self.states = self.dynamics.propagate(self.states, self._pending_actions)
        self.step_count += 1

        p = self.states[:, 0:3]
        w = self.states[:, 10:13]

        gate_R   = self.gates_R[self.gate_idx]
        gate_pos = self.gates_pos[self.gate_idx]
        RT       = gate_R.transpose(-1, -2)

        prev_rel = (RT @ (prev_pos - gate_pos).unsqueeze(-1)).squeeze(-1)  # (N, 3)
        curr_rel = (RT @ (p       - gate_pos).unsqueeze(-1)).squeeze(-1)   # (N, 3)

        # progress      = curr_rel[:, 1] - prev_rel[:, 1]
        progress      =  prev_rel.norm(dim=-1) - curr_rel.norm(dim=-1)  
        plane_crossed = (prev_rel[:, 1] < 0.0) & (curr_rel[:, 1] >= 0.0)
        inside_gate   = (curr_rel[:, 0].abs() < GATE_HALF_W) & (curr_rel[:, 2].abs() < GATE_HALF_H)
        gate_success  = plane_crossed & inside_gate
        gate_crash    = plane_crossed & ~inside_gate

        if gate_success.any():
            self.gate_idx[gate_success] = (self.gate_idx[gate_success] + 1) % self.n_gates
            # progress[gate_success]      = 0.0

        terminated = (
            (p[:, 0].abs() > X_THRESH) |
            (p[:, 1].abs() > Y_THRESH) |
            (p[:, 2] < 0.0)            |
            (p[:, 2] > Z_THRESH)       |
            gate_crash
        )
        truncated = (self.step_count >= MAX_STEPS) | gate_success 
        dones     = terminated | truncated

        rewards = (
              1.0  * progress
            - 0.001 * w.norm(dim=-1)
            # - 10.0  * terminated.float()
        )

        obs      = self._get_obs()
        rew_np   = rewards.cpu().numpy().astype(np.float32)
        done_np  = dones.cpu().numpy()

        self.episode_returns += rewards
        self.episode_lengths += 1

        infos: List[dict] = [{} for _ in range(self.num_envs)]
        if dones.any():
            done_idx = dones.nonzero(as_tuple=True)[0].tolist()
            for i in done_idx:
                infos[i]["terminal_observation"] = obs[i].copy()
                if truncated[i] and not terminated[i]:
                    infos[i]["TimeLimit.truncated"] = True
                infos[i]["episode"] = {
                    "r": self.episode_returns[i].item(),
                    "l": self.episode_lengths[i].item(),
                }
            self.episode_returns[done_idx] = 0
            self.episode_lengths[done_idx] = 0
            self._reset_envs(dones)
            new_obs = self._get_obs()
            obs[done_idx] = new_obs[done_idx]

        return obs, rew_np, done_np, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> list:
        n = len(self._get_indices(indices))
        return [getattr(self, attr_name)] * n

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list:
        n = len(self._get_indices(indices))
        result = getattr(self, method_name)(*method_args, **method_kwargs)
        return [result] * n

    def env_is_wrapped(self, wrapper_class, indices=None) -> list:
        return [False] * len(self._get_indices(indices))

    def seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return [seed] * self.num_envs


# ─────────────────────────────────────────────────────────────────────────────
# Single-env gymnasium wrapper  (inference / rendering via test.py)
# ─────────────────────────────────────────────────────────────────────────────

class QuadrotorEnv(gym.Env):
    """
    Standard single-env gymnasium interface backed by the same PyTorch dynamics
    (N=1, CPU).  Keeps the renderer path from test.py working unchanged.

    State: [p(3) | v(3) | q(4) | ω(3)]
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, log_dir: Optional[str] = PPO_LOG_DIR):
        super().__init__()
        self.render_mode = render_mode
        self._renderer   = None
        self._stop_simulation = False
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._log_file = None
        self._log_writer = None
        self._episode_id = -1
        self._step_index = 0

        gates_pos_np, gates_rpy_np = load_gates_from_yaml(TRACK_PATH)
        self.gates_position = gates_pos_np   # (G, 3) numpy, kept for renderer
        self.gates_rpy      = gates_rpy_np   # (G, 3) numpy

        self.dynamics = QuadrotorDynamics(
            mass=MASS, inertia=INERTIA, length=LENGTH, gravity=GRAVITY,
            dt=DT, max_thrust=MAX_T, torque_const=TORQUE_C, device="cpu",
        )

        obs_high = np.array([
            X_THRESH*2, Y_THRESH*2, Z_THRESH*2,
            40., 40., 40.,
            1., 1., 1., 1.,
            np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
            1., 1., 1.,
            float(len(self.gates_position)),
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)

        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = self._log_dir / f"quadrotor_state_log_{timestamp}.csv"
            self._log_file = open(log_path, "w", newline="", encoding="utf-8")
            fieldnames = [
                "episode",
                "step",
                "time_s",
                "gate_idx",
                "reward",
                "terminated",
                "truncated",
            ] + [f"state_{i}" for i in range(13)] + [f"action_{i}" for i in range(4)]
            self._log_writer = csv.DictWriter(self._log_file, fieldnames=fieldnames)
            self._log_writer.writeheader()

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_id += 1
        self._step_index = 0

        n_gates = len(self.gates_position)
        self.gate_idx        = int(self.np_random.integers(n_gates))
        self._curr_gate_pos  = self.gates_position[self.gate_idx]
        self._curr_gate_rpy  = self.gates_rpy[self.gate_idx]

        gate_R  = rpy_to_rotmat_np(self._curr_gate_rpy)
        offset  = np.array([0., -1., 0.], dtype=np.float32)
        noise   = np.array([
            self.np_random.normal(0, 0.2),
            self.np_random.normal(0, 0.2),
            self.np_random.normal(0, 0.2),
        ], dtype=np.float32)

        self._state = np.zeros(13, dtype=np.float32)
        self._state[0:3] = self._curr_gate_pos + gate_R @ (offset + noise)
        # self._state[2] = 1.0
        self._state[3:6] = self.np_random.uniform(-0.5, 0.5, 3).astype(np.float32)
        self._state[6]   = 1.0   # identity quaternion

        self._prev_pos   = self._state[0:3].copy()
        self._step_count = 0
        self._stop_simulation = False
        self._lap_start_gate_idx = self.gate_idx
        self._lap_start_step = 0
        self._lap_started = False
        self._lap_count = 0

        if self._renderer is not None:
            self._renderer.set_target(self.gate_idx)

        return self._get_obs(), {}

    def _log_step(self, action: np.ndarray, reward: float, terminated: bool, truncated: bool) -> None:
        if self._log_writer is None:
            return

        row = {
            "episode": self._episode_id,
            "step": self._step_index,
            "time_s": float(self._step_index * DT),
            "gate_idx": int(self.gate_idx),
            "reward": float(reward),
            "terminated": int(bool(terminated)),
            "truncated": int(bool(truncated)),
        }
        row.update({f"state_{i}": float(value) for i, value in enumerate(self._state.tolist())})
        row.update({f"action_{i}": float(value) for i, value in enumerate(np.asarray(action, dtype=np.float32).tolist())})
        self._log_writer.writerow(row)
        self._log_file.flush()
        self._step_index += 1

    def step(self, action: np.ndarray):
        if self._stop_simulation:
            raise SystemExit("Simulation closed (x pressed)")

        if self._renderer and self._renderer._paused:
            self.render()
            return self._get_obs(), 0.0, False, False, {}

        # PyTorch propagation (N=1, CPU)
        state_t  = torch.from_numpy(self._state).unsqueeze(0)   # (1, 13)
        action_t = torch.from_numpy(action.astype(np.float32)).unsqueeze(0)  # (1, 4)
        self._state = self.dynamics.propagate(state_t, action_t)[0].numpy()

        self._step_count += 1
        p = self._state[0:3]
        w = self._state[10:13]

        # Gate crossing
        gate_R = rpy_to_rotmat_np(self._curr_gate_rpy)
        RT     = gate_R.T
        prev_rel = RT @ (self._prev_pos - self._curr_gate_pos)
        curr_rel = RT @ (p              - self._curr_gate_pos)

        progress     = curr_rel[1] - prev_rel[1]
        plane_crossed = prev_rel[1] < 0.0 and curr_rel[1] >= 0.0
        inside_gate   = abs(curr_rel[0]) < GATE_HALF_W and abs(curr_rel[2]) < GATE_HALF_H
        gate_success  = plane_crossed and inside_gate
        gate_crash    = plane_crossed and not inside_gate

        if gate_success:
            if not self._lap_started:
                self._lap_started = True
                self._lap_start_gate_idx = self.gate_idx
                self._lap_start_step = self._step_count
            elif self.gate_idx == self._lap_start_gate_idx:
                self._lap_count += 1
                lap_time = (self._step_count - self._lap_start_step) * DT
                print(f"Lap {self._lap_count}: {lap_time:.2f} s")
                self._lap_start_step = self._step_count
            self.gate_idx       = (self.gate_idx + 1) % len(self.gates_position)
            self._curr_gate_pos = self.gates_position[self.gate_idx]
            self._curr_gate_rpy = self.gates_rpy[self.gate_idx]
            # progress            = 0.0
            
            if self._renderer is not None:
                self._renderer.set_target(self.gate_idx)

        terminated = bool(
               p[0] < -X_THRESH or p[0] > X_THRESH
            or p[1] < -Y_THRESH or p[1] > Y_THRESH
            or p[2] < 0.0 or p[2] > Z_THRESH
            or gate_crash
        )
        truncated = self._step_count >= MAX_STEPS #or gate_success

        reward = (
              1.0  * progress
            - 0.001 * np.linalg.norm(w)
            # - 10.0  * terminated
        )

        self._log_step(action=action, reward=float(reward), terminated=terminated, truncated=truncated)

        self._prev_pos = p.copy()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        p = self._state[0:3]
        v = self._state[3:6]
        q = self._state[6:10]
        w = self._state[10:13]

        gate_R   = rpy_to_rotmat_np(self._curr_gate_rpy)
        RT       = gate_R.T
        rel_pos  = RT @ (p - self._curr_gate_pos)
        rel_vel  = RT @ v
        gate_norm = gate_R[:, 1]                    # y-axis = forward through gate

        return np.concatenate([
            rel_pos, rel_vel, q, w, gate_norm,
            np.array([self.gate_idx], dtype=np.float32),
        ]).astype(np.float32)

    def render(self):
        if self.render_mode is None:
            return
        if self._renderer is None:
            from utils.renderer import RacingRenderer
            self._renderer = RacingRenderer(
                gates_position=self.gates_position,
                gates_rpy=self.gates_rpy,
                gate_mesh_path="misc/gate.obj",
                arm_length=LENGTH,
                arm_angle=45.0,
                mass=MASS,
                dt=DT,
            )
        alive = self._renderer.step(self._state)
        if not alive:
            self._stop_simulation = True
            self.render_mode = None

    def close(self):
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
            self._log_writer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
