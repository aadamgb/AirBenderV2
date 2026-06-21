import torch
from utils.randomizer import QuadrotorRandomizer
from dynamics.quadrotor_dynamics import QuadrotorDynamics
from utils.renderer import BaseRenderer

num_envs = 3
device = "cuda"

randomizer = QuadrotorRandomizer(
    inertia=(0.1,0.1,0.5),
    length=0.2,
    torque_c=0.012,
    angle=45.0,
)
dyanamics = QuadrotorDynamics(
    *randomizer.sample(num_envs, randomize=False),
    gravity=9.81, max_thrust=20.0, dt=0.01,
)

actions = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device).expand(num_envs, -1)
for step in range(2):
    if step == 1:
        dyanamics.set_params(*randomizer.sample(2), idx=[0,2])
    print(f"✅ PARAMS FOR STEP: {step+1} ✅")   
    dyanamics.print_params()
