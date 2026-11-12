## Installation Train

### Option A: pixi (Isaac Lab 3.0 release, recommended)

1. Install [pixi](https://pixi.prefix.dev/latest/installation/) and git-lfs (`sudo apt install git-lfs`), then clone this repository.

2. From anywhere inside the repo:

```bash
pixi shell -e train
```

This creates an environment with Python 3.12 and `isaaclab[all,wandb]==3.0.0` from PyPI (Newton backend, no Isaac Sim), and installs this repo in editable mode. It requires Ubuntu 22.04 or newer (glibc >= 2.35).

If you change the dependencies in `source/position_based_locomotion_isaaclab/setup.py`, run `pixi update -e train position-based-locomotion-isaaclab`.

### Option B: your own Isaac Lab installation

1. Install Isaac Lab by following the [installation guide](https://github.com/isaac-sim/IsaacLab).

2. Install git for very large file
```bash
sudo apt install git-lfs
```

3. Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory)


4. Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/position_based_locomotion_isaaclab
```

### Extensions

- If you want to play with [Morphologycal Symmetries](https://arxiv.org/pdf/2403.17320), install the repo [morphosymm-rl](https://github.com/iit-DLSLab/morphosymm-rl)


## Run a train/play in IsaacLab

- To train:

```bash
python scripts/rsl_rl/train.py --task=PositionBased-Go2-Flat --num_envs=4096
python scripts/rsl_rl/train.py --task=PositionBased-Go2-Rough-Blind --num_envs=4096
python scripts/rsl_rl/train.py --task=PositionBased-Go2-Rough-Vision --num_envs=4096
```

- To test the policy, you can press:
```bash
python scripts/rsl_rl/play.py --task=PositionBased-Go2-Flat --num_envs=16 --visualizer newton
python scripts/rsl_rl/play.py --task=PositionBased-Go2-Rough-Blind --num_envs=16 --visualizer newton
```

Newton is the default physics backend; add `physics=physx` to use PhysX.

`PositionBased-Go2-Rough-Vision` also declares a D435 depth camera, used by [iekf-quadruped-isaaclab](https://github.com/iit-DLSLab/iekf-quadruped-isaaclab) to collect visual odometry data. It stays disabled during the policy training.


## Use Morphological Symmetries
The module has its own README in its folder under [scripts](./scripts/morphosymm_rl/README_morphosymm.md).
