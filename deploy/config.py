"""Robot and policy configuration for deployment."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

robot = "go2"
scene = "scene_random_boxes"  # scene_flat, scene_random_boxes

locomotion_policy_folder = PROJECT_ROOT / "tested_policies" / robot /"symm"
locomotion_policy_folder_path = str(locomotion_policy_folder)

Kp_walking = 20.0
Kd_walking = 1.5

Kp_stand_up_and_down = 50.0
Kd_stand_up_and_down = 5.0

# Load the parameters serialized with the selected policy.
with (locomotion_policy_folder / "params" / "env.yaml").open("r", encoding="utf-8") as file:
    training_env = yaml.unsafe_load(file)

# Keep the checked-in policies, serialized before the scanner rename, deployable.
if "perceptive_height_scanner" not in training_env and "height_scanner2" in training_env:
    training_env["perceptive_height_scanner"] = training_env["height_scanner2"]

concurrent_state_est_network = str(locomotion_policy_folder / "exported" / "concurrent_state_estimator.pth")
rma_network = str(locomotion_policy_folder / "exported" / "rma.pth")
