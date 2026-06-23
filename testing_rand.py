import torch
from utils.randomizer import QuadrotorRandomizer
from dynamics.quadrotor_dynamics import QuadrotorDynamics
from utils.renderer import BaseRenderer

num_envs = 2
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
for step in range(1):
    if step == 0:
        dyanamics.set_params(*randomizer.sample(1), idx=[0])
    # print(f"\n ✅ PARAMS FOR STEP: {step+1} ✅")   
    # dyanamics.print_params()
    # dyanamics._compute_drag(torch.tensor([1.0, 0.0, 0.0], device=device), torch.tensor([[0.9659, 0.0, 0.0, 0.2588]], device=device))
    dyanamics._motor_dynamics(Omega=torch.tensor([[100000.0, 100000.0, 100000.0, 100000.0], [0.0, 0.0, 0.0, 0.0]], device=device),thrust_cmd= torch.tensor([[0.0, 0.0, 0.0, 0.0], [25.0, 25.0, 25.0, 25.0]], device=device))
