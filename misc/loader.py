import yaml
import numpy as np

def load_gates_from_yaml(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads gate positions and RPY angles from a YAML file.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    gates = data["gates"]
    positions = np.array([g["position"] for g in gates], dtype=np.float32)
    rpys      = np.array([g["rpy"]      for g in gates], dtype=np.float32)
    return positions, rpys