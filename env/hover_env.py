import torch
import os
from torch import nn
from collections import deque
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

X_THRESH        = 5.0
Y_THRESH        = 8.0
Z_THRESH        = 3.0
GATE_HALF_W     = 0.6
GATE_HALF_H     = 0.6

EPISODES    = 100
STEPS       = 1200
HORIZON     = 50

TRACK_PATH = "../misc/racing_tracks/fig8.yaml"

num_envs = 100
device = "cuda"

gates_pos_np, gates_rpy_np = load_gates_from_yaml(TRACK_PATH)

gates_pos_t   = torch.tensor(gates_pos_np, dtype=torch.float32, device=device)
gates_rpy_t   = torch.tensor(gates_rpy_np, dtype=torch.float32, device=device)
gate_idx      = torch.zeros((num_envs,), dtype=torch.long, device=device)
gates_R_t     = rpy_to_rotmat(gates_rpy_t)
n_gates       = len(gates_pos_np)

policy = MLP(
    layer_sizes=[17, 64, 64, 64, 4],
    activation=nn.ReLU,
    output_activation=nn.Sigmoid,
).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

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

states = torch.zeros((num_envs, 17), device=device)
total_timesteps = 0
ep_rew_buffer = deque(maxlen=100)  # mirrors SB3's ep_info_buffer


def get_obs(states_t, gate_idx_t):
    p = states_t[:, 0:3]
    v = states_t[:, 3:6]
    q = states_t[:, 6:10]
    w = states_t[:, 10:13]

    gate_R = gates_R_t[gate_idx_t]
    RT = gate_R.transpose(-1, -2)
    rel_pos = (RT @ (p - gates_pos_t[gate_idx_t]).unsqueeze(-1)).squeeze(-1)
    rel_vel = (RT @ v.unsqueeze(-1)).squeeze(-1)
    gate_norm = gate_R[:, :, 1]
    gate_idx_f = gate_idx_t.to(dtype=states_t.dtype).unsqueeze(-1)

    return torch.cat([rel_pos, rel_vel, q, w, gate_norm, gate_idx_f], dim=-1)


def reset(n):
    target_gate = torch.randint(0, n_gates, (n,), device=device)
    gate_R   = gates_R_t[target_gate]
    gate_pos = gates_pos_t[target_gate]

    local_offset = torch.zeros((n, 3), device=device)
    local_offset[:, 0] = (torch.rand(n, device=device) * 2 - 1) * 0.5     # lateral jitter
    local_offset[:, 1] = -(torch.rand(n, device=device) * 1.5 + 0.5)      # 0.5-2.0 m behind the gate
    local_offset[:, 2] = (torch.rand(n, device=device) * 2 - 1) * 0.5     # vertical jitter

    world_offset = (gate_R @ local_offset.unsqueeze(-1)).squeeze(-1)

    state = torch.zeros((n, 13), device=device)
    state[:, 0:3] = gate_pos + world_offset
    state[:, 3:6] = torch.rand((n, 3), device=device) * 2 - 1
    state[:, 6]   = 1.0
    return state, target_gate

for ep in range(EPISODES):
    # states = reset(num_envs)
    states, gate_idx = reset(num_envs)
    # gate_idx = torch.zeros((num_envs,), dtype=torch.long, device=device)

    optimizer.zero_grad()
    loss = torch.zeros(1, device=device)
    episode_returns = torch.zeros(num_envs, device=device)

    for t in range(STEPS):
        prev_p = states[:, 0:3].clone()

        obs = get_obs(states, gate_idx)
        actions = policy(obs)

        states = dynamics.propagate(states, actions)

        p, w = states[:, 0:3], states[:, 10:13]

        RT = gates_R_t[gate_idx].transpose(-1, -2)
        prev_rel = (RT @ (prev_p - gates_pos_t[gate_idx]).unsqueeze(-1)).squeeze(-1)
        curr_rel = (RT @ (p - gates_pos_t[gate_idx]).unsqueeze(-1)).squeeze(-1)

        progress = prev_rel.norm(dim=-1) - curr_rel.norm(dim=-1)
        reward   = progress - 0.001 * w.norm(dim=-1)

        episode_returns += reward.detach()
        total_timesteps += num_envs
        loss -= reward.mean()

        plane_crossed = (prev_rel[:, 1] < 0.0) & (curr_rel[:, 1] >= 0.0)
        inside_gate   = (curr_rel[:, 0].abs() < GATE_HALF_W) & (curr_rel[:, 2].abs() < GATE_HALF_H)
        gate_success  = plane_crossed & inside_gate
        gate_crash    = plane_crossed & ~inside_gate

        lap_complete  = gate_success & (gate_idx == n_gates - 1) 

        if gate_success.any():
            gate_idx[gate_success] = (gate_idx[gate_success] + 1) % n_gates
            if renderer is not None:
                renderer.set_target(gate_idx[0].cpu().numpy())

        terminated = (
            (p[:, 0].abs() > X_THRESH) |
            (p[:, 1].abs() > Y_THRESH) |
            (p[:, 2] < 0.0)            |
            (p[:, 2] > Z_THRESH)       |
            gate_crash
        )
        truncated = lap_complete
        dones = terminated #| truncated

        if dones.any():
            done_idx = dones.nonzero(as_tuple=True)[0]
            ep_rew_buffer.extend(episode_returns[done_idx].tolist())
            new_states_subset, new_gate_subset = reset(len(done_idx))
            new_states = states.clone()
            new_states[done_idx] = new_states_subset
            states = new_states

            gate_idx[done_idx] = new_gate_subset
            episode_returns[done_idx] = 0.0

        if ep == EPISODES - 1:
            renderer.step(states[0].detach().cpu().numpy())

        # --- optimize every HORIZON steps (truncated BPTT) ---
        is_chunk_end = (t + 1) % HORIZON == 0
        is_last_step = t == STEPS - 1
        if is_chunk_end or is_last_step:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            states = states.detach()

            loss = torch.zeros(1, device=device)

    ep_rew_mean = sum(ep_rew_buffer) / len(ep_rew_buffer) if ep_rew_buffer else float("nan")
    print(f"Episode {ep}: timesteps = {total_timesteps}, ep_rew_mean = {ep_rew_mean:.4f}")


torch.save(policy.state_dict(), "../outputs/bptt/policy_bptt.pt")