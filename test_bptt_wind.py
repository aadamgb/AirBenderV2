import time
import csv
from datetime import datetime
from pathlib import Path

import torch

from utils.nn import MLP
from utils.math import rpy_to_rotmat
from utils.renderer import RacingRenderer
from misc.loader import load_gates_from_yaml
from dynamics.quadrotor_dynamics import QuadrotorDynamics

GRAVITY   = 9.81
MASS      = 1.21
INERTIA   = (0.007, 0.007, 0.013)
LENGTH    = 0.15
MAX_T     = 20.0
DT        = 0.01
TORQUE_C  = 0.012

X_THRESH        = 10.0
Y_THRESH        = 10.0
Z_THRESH        = 10.0
GATE_HALF_W     = 0.6
GATE_HALF_H     = 0.6

Y_WIND_MIN = -1.0
Y_WIND_MAX =  1.0
DIST_X_DIR = 3.0 # m/s2

TRACK_PATH  = "misc/racing_tracks/fig8.yaml"
POLICY_PATH = "outputs/bptt/fig8.pt"   
LOG_DIR     = Path("/home/adame/AirBender/outputs/bptt_logs")

device = "cuda"

gates_pos_np, gates_rpy_np = load_gates_from_yaml(TRACK_PATH)
gates_pos_t = torch.tensor(gates_pos_np, dtype=torch.float32, device=device)
gates_rpy_t = torch.tensor(gates_rpy_np, dtype=torch.float32, device=device)
gates_R_t   = rpy_to_rotmat(gates_rpy_t)
n_gates     = len(gates_pos_np)

policy = MLP(
    layer_sizes=[20, 64, 64, 64, 4],
    activation=torch.nn.ReLU,
    output_activation=torch.nn.Sigmoid,
).to(device)
policy.load_state_dict(torch.load(POLICY_PATH, map_location=device))
policy.eval()

dynamics = QuadrotorDynamics(
    mass=MASS,
    inertia=INERTIA,
    length=LENGTH,
    gravity=GRAVITY,
    dt=DT,
    max_thrust=MAX_T,
    torque_const=TORQUE_C,
)

renderer = RacingRenderer(gates_pos_np, gates_rpy_np)

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR / f"quadrotor_state_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_file = open(log_path, "w", newline="", encoding="utf-8")
log_fieldnames = [
    "episode",
    "step",
    "time_s",
    "gate_idx",
    "reward",
    "terminated",
    "truncated",
] + [f"state_{i}" for i in range(13)] + [f"action_{i}" for i in range(4)]
log_writer = csv.DictWriter(log_file, fieldnames=log_fieldnames)
log_writer.writeheader()


def get_obs(states_t, gate_idx_t):
    p = states_t[:, 0:3]
    v = states_t[:, 3:6]
    q = states_t[:, 6:10]
    w = states_t[:, 10:13]

    gate_pos = gates_pos_t[gate_idx_t]
    gate_R = gates_R_t[gate_idx_t]
    RT = gate_R.transpose(-1, -2)
    rel_pos = (RT @ (p - gate_pos).unsqueeze(-1)).squeeze(-1)
    rel_vel = (RT @ v.unsqueeze(-1)).squeeze(-1)
    gate_norm = gate_R[:, :, 1]
    gate_idx_f = gate_idx_t.to(dtype=states_t.dtype).unsqueeze(-1)

    return torch.cat([rel_pos, rel_vel, q, w, gate_pos, gate_norm, gate_idx_f], dim=-1)


def sample_init(n):
    init = torch.zeros((n, 13), device=device)
    init[:, 0:6] = torch.rand((n, 6), device=device) * 2 - 1
    init[:, 2] = 1.5
    init[:, 6] = 1.0
    return init

def reset(n):
    # target_gate = torch.randint(0, n_gates, (n,), device=device)
    target_gate = torch.zeros((n,), dtype=torch.long, device=device)
    gate_R   = gates_R_t[target_gate]
    gate_pos = gates_pos_t[target_gate]

    local_offset = torch.zeros((n, 3), device=device)
    local_offset[:, 0] = (torch.rand(n, device=device) * 2 - 1) * 0.5     # lateral jitter
    local_offset[:, 1] = -(torch.rand(n, device=device) * 1.5 + 0.5)      # 0.5-2.0 m behind the gate
    local_offset[:, 2] = (torch.rand(n, device=device) * 2 - 1) * 0.5     # vertical jitter

    world_offset = (gate_R @ local_offset.unsqueeze(-1)).squeeze(-1)

    state = torch.zeros((n, 13), device=device)
    state[:, 2] = 1.5
    state[:, 0:3] = gate_pos + world_offset
    # state[:, 3:6] = torch.rand((n, 3), device=device) * 2 - 1
    state[:, 6]   = 1.0
    return state, target_gate

state, gate_idx    = reset(1)
# gate_idx = torch.zeros((1,), dtype=torch.long, device=device)

episode   = 0
ep_reward = 0.0
ep_steps  = 0
quit_requested = False
lap_started = False
lap_start_gate_idx = None
lap_start_step = 0
lap_count = 0


def log_step(action, reward, terminated, truncated):
    row = {
        "episode": episode,
        "step": ep_steps,
        "time_s": float(ep_steps * DT),
        "gate_idx": int(gate_idx[0].item()),
        "reward": float(reward),
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
    }
    row.update({f"state_{i}": float(value) for i, value in enumerate(state[0].tolist())})
    row.update({f"action_{i}": float(value) for i, value in enumerate(action[0].tolist())})
    log_writer.writerow(row)
    log_file.flush()

with torch.no_grad():
    while True:
        while renderer.is_paused:
            if not renderer.step(state[0].cpu().numpy()):
                quit_requested = True
                break
            time.sleep(0.02)

        if quit_requested:
            break
        if renderer is not None:
                renderer.set_target(gate_idx[0].cpu().numpy())

        prev_p = state[:, 0:3].clone()

        obs    = get_obs(state, gate_idx)
        action = policy(obs)  
        state  = dynamics.propagate(state, action)

        p, w = state[:, 0:3], state[:, 10:13]

        # Adding wind disturbance
        in_zone = (p[:, 1] >= Y_WIND_MIN) & (p[:, 1] <= Y_WIND_MAX)
        # print(in_zone)
        state[in_zone, 3 ] -= DIST_X_DIR * DT 

        RT       = gates_R_t[gate_idx].transpose(-1, -2)
        prev_rel = (RT @ (prev_p - gates_pos_t[gate_idx]).unsqueeze(-1)).squeeze(-1)
        curr_rel = (RT @ (p       - gates_pos_t[gate_idx]).unsqueeze(-1)).squeeze(-1)

        progress = prev_rel.norm(dim=-1) - curr_rel.norm(dim=-1)
        reward   = (progress - 0.001 * w.norm(dim=-1)).item()

        plane_crossed = (prev_rel[:, 1] < 0.0) & (curr_rel[:, 1] >= 0.0)
        inside_gate   = (curr_rel[:, 0].abs() < GATE_HALF_W) & (curr_rel[:, 2].abs() < GATE_HALF_H)
        gate_success  = plane_crossed & inside_gate
        gate_crash    = plane_crossed & ~inside_gate

        if gate_success.any():
            if not lap_started:
                lap_started = True
                lap_start_gate_idx = int(gate_idx[0].item())
                lap_start_step = ep_steps
            elif int(gate_idx[0].item()) == lap_start_gate_idx:
                lap_count += 1
                lap_time = (ep_steps - lap_start_step) * DT
                print(f"Lap {lap_count}: {lap_time:.2f} s")
                lap_start_step = ep_steps
            gate_idx[gate_success] = (gate_idx[gate_success] + 1) % n_gates
            curr_gate_pos = gates_pos_t[gate_idx]
            if renderer is not None:
                renderer.set_target(gate_idx[0].cpu().numpy())

        terminated = bool((
            (p[:, 0].abs() > X_THRESH) |
            (p[:, 1].abs() > Y_THRESH) |
            (p[:, 2] < 0.0)            |
            (p[:, 2] > Z_THRESH)       |
            gate_crash
        ).item())
        truncated = bool(gate_success.item())

        ep_reward += reward
        ep_steps  += 1
        log_step(action=action, reward=reward, terminated=terminated, truncated=truncated)

        if not renderer.step(state[0].cpu().numpy()):
            break

        if terminated: # or truncated:
            episode += 1
            print(f"Episode {episode:3d} | steps: {ep_steps:4d} | reward: {ep_reward:.1f}")
            ep_reward = 0.0
            ep_steps  = 0
            state, gate_idx    = reset(1)
            # gate_idx = torch.zeros((1,), dtype=torch.long, device=device)
            time.sleep(0.3)

log_file.close()