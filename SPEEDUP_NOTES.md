# Training Speed Optimizations

## 1. Fuse the inner physics loop with `torch.compile`

The hot path in `step_wait` launches a separate batch of CUDA kernels for every `propagate` call.
Since `n_int_steps = 4` is a compile-time constant, compile a fused 4-step version so all four RK4
passes get traced into one kernel graph.

**In `QuadrotorVecEnv.__init__`:**
```python
N_INT = round(DT_RATE / DT_INT)  # = 4

def _propagate_n(dynamics, n):
    def fn(state, cmd):
        for _ in range(n):
            state = dynamics.propagate(state, cmd)
        return state
    return torch.compile(fn)

self._compiled_propagate = _propagate_n(self.dynamics, N_INT)
```

**In `step_wait`, replace the inner loop:**
```python
for _ in range(n_rate_steps):
    thrust_commands = self.controller.map(self.states, self._pending_actions, dt=DT_RATE)
    self.states = self._compiled_propagate(self.states, thrust_commands)
```

`torch.compile` traces through `for _ in range(4)` and fuses it — the loop disappears at runtime.

---

## 2. Remove the double `_get_obs()` call on episode done

When any env resets, `_get_obs()` is called twice (lines 241 and 263), causing two GPU→CPU syncs.
Call it once after the reset and index into it directly.

**Current (bad):**
```python
obs = self._get_obs()           # sync 1
...
if dones.any():
    ...
    self._reset_envs(dones)
    new_obs = self._get_obs()   # sync 2 (redundant)
    obs[done_idx] = new_obs[done_idx]
```

**Fix:** move the first `_get_obs()` call to after the reset block, or only call it once at the end.

---

## 3. Merge `dones.any()` with `done_np`

`dones.any()` on its own is a GPU→CPU sync (reads a scalar back to Python).
You already have `done_np = dones.cpu().numpy()`, so reuse it:

```python
# replace:
if dones.any():

# with:
if done_np.any():
```

---

## 4. Increase `num_envs`

With N=100 and a 17-dim state, the GPU is likely underutilized — tensors are too small to saturate
CUDA cores. Bump to 512 or 1024 in `train.py`:

```python
env = QuadrotorVecEnv(num_envs=512, ...)
```

SB3 already collects `num_envs × n_steps` samples per update, so larger batches come for free.

---

## 5. Compile `controller.map`

If the controller is pure tensor ops, wrap it the same way:

```python
self._compiled_controller_map = torch.compile(self.controller.map)
```

Then use `self._compiled_controller_map(...)` in `step_wait`.

---

## Summary

| Fix | File | Effort | Expected gain |
|---|---|---|---|
| Compile fused 4-step propagate | `env/quadrotor_env.py` | Low | High — removes 3/4 of kernel launches |
| Remove double `_get_obs()` on done | `env/quadrotor_env.py` | Trivial | Medium |
| Use `done_np.any()` instead of `dones.any()` | `env/quadrotor_env.py` | Trivial | Small |
| Increase `num_envs` to 512–1024 | `train.py` | Trivial | High if GPU underutilized |
| Compile `controller.map` | `env/quadrotor_env.py` | Low | Medium |
