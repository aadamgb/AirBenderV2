import time
import torch
from dynamics.quadrotor_dynamics import QuadrotorDynamics
from utils.randomizer import QuadrotorRandomizer

device = "cuda"
params = QuadrotorRandomizer()
quad = QuadrotorDynamics(*params.sample(num_envs=1, randomize=False))

action = torch.tensor([[10.0, 10.0, 10.0, 10.0]], device=device)

N_WARMUP = 100
N_ITERS  = 20000

def fresh_state():
    s = torch.zeros((1, 17), device=device)
    s[0, 6] = 1.0
    return s

# ── Uncompiled ────────────────────────────────────────────────────────────────
state = fresh_state()
for _ in range(N_WARMUP):
    state = quad.propagate(state, action)
torch.cuda.synchronize()

state = fresh_state()
t0 = time.perf_counter()
for _ in range(N_ITERS):
    state = quad.propagate(state, action)
torch.cuda.synchronize()
t_base = time.perf_counter() - t0

print(f"[Uncompiled]  {N_ITERS} iters | total {t_base*1e3:.2f} ms | {t_base/N_ITERS*1e6:.2f} µs/iter")

# ── torch.compile ─────────────────────────────────────────────────────────────
compiled_propagate = torch.compile(quad.propagate)

state = fresh_state()
for _ in range(N_WARMUP):                  # first call triggers compilation
    state = compiled_propagate(state, action)
torch.cuda.synchronize()

state = fresh_state()
t0 = time.perf_counter()
for _ in range(N_ITERS):
    state = compiled_propagate(state, action)
torch.cuda.synchronize()
t_compiled = time.perf_counter() - t0

print(f"[Compiled]    {N_ITERS} iters | total {t_compiled*1e3:.2f} ms | {t_compiled/N_ITERS*1e6:.2f} µs/iter")

print(f"\nSpeedup: {t_base/t_compiled:.2f}x  ({'faster' if t_compiled < t_base else 'slower'} with compile)")
