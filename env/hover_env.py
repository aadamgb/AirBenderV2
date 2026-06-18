import torch
from torch import nn
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

X_THRESH        = 4.0
Y_THRESH        = 8.0
Z_THRESH        = 3.0
GATE_HALF_W     = 0.6
GATE_HALF_H     = 0.6

EPISODES    = 10
STEPS       = 200
HORIZON     = 100      

TRACK_PATH = "../misc/racing_tracks/one_gate.yaml"

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

states = torch.zeros((num_envs, 13), device=device)


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


for ep in range(EPISODES):
    states = torch.zeros((num_envs, 13), device=device)
    states[:, 0:6] = torch.rand((num_envs, 6), device=device) * 2 - 1
    states[:, 2] = 1.5
    states[:, 6] = 1.0
    gate_idx = torch.zeros((num_envs,), dtype=torch.long, device=device)

    optimizer.zero_grad()
    loss = torch.zeros(1, device=device)
    ep_loss_sum = 0.0  # accumulates across all chunks, for logging only

    for t in range(STEPS):
        prev_p = states[:, 0:3].clone()

        obs = get_obs(states, gate_idx)
        actions = policy(obs)

        states = dynamics.propagate(states, actions)

        p = states[:, 0:3]

        RT = gates_R_t[gate_idx].transpose(-1, -2)
        prev_rel = (RT @ (prev_p - gates_pos_t[gate_idx]).unsqueeze(-1)).squeeze(-1)
        curr_rel = (RT @ (p - gates_pos_t[gate_idx]).unsqueeze(-1)).squeeze(-1)
        progress = curr_rel.norm(dim=-1) - prev_rel.norm(dim=-1)

        plane_crossed = (prev_rel[:, 1] < 0.0) & (curr_rel[:, 1] >= 0.0)
        inside_gate   = (curr_rel[:, 0].abs() < GATE_HALF_W) & (curr_rel[:, 2].abs() < GATE_HALF_H)
        gate_success  = plane_crossed & inside_gate
        gate_crash    = plane_crossed & ~inside_gate

        step_loss = progress.mean() + 0.001 * (states[:, 10:13] ** 2).sum(dim=-1).mean()
        loss += step_loss

        if gate_success.any():
            gate_idx[gate_success] = (gate_idx[gate_success] + 1) % n_gates

        # if ep == EPISODES - 1:
        if (ep+1) % 10 == 0:
            renderer.step(states[0].detach().cpu().numpy())

        # --- optimize every OPT_EVERY steps (truncated BPTT) ---
        is_chunk_end = (t + 1) % HORIZON == 0
        is_last_step = t == STEPS - 1
        if is_chunk_end or is_last_step:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            states = states.detach()
            ep_loss_sum += loss.item()

            loss = torch.zeros(1, device=device)

    print(f"Episode {ep}: loss={ep_loss_sum:.4f}")