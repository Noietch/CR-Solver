<div align="center">

<h1>CR-Solver: GPU-Accelerated Kinematics Solver for Tendon-driven Continuum Robots</h1>

<div>
    <a href='https://github.com/Noietch/DiffSoft' target='_blank'>Heqing Yang</a><sup>1</sup>&emsp;
    <a href='https://github.com/Noietch/DiffSoft' target='_blank'>Yang Yi</a><sup>1</sup>&emsp;
    <a href='https://github.com/Noietch/DiffSoft' target='_blank'>Linqing Zhong</a><sup>1</sup>&emsp;
    <a href='https://github.com/Noietch/DiffSoft' target='_blank'>Linjiang Huang</a><sup>1†</sup>&emsp;
    <a href='https://github.com/Noietch/DiffSoft' target='_blank'>Si Liu</a><sup>1†</sup>
</div>
<div>
    <sup>1</sup>Beihang University&emsp;<sup>†</sup>Corresponding authors
</div>

<div>
    <strong>IROS 2026</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://github.com/Noietch/DiffSoft" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg">
        </a>
        <a href="https://github.com/Noietch/DiffSoft" target='_blank'>
        <img src="https://img.shields.io/badge/Project-Page-green">
        </a>
        <a href="https://github.com/Noietch/DiffSoft" target='_blank'>
        <img src="https://img.shields.io/badge/Cite-BibTeX-blue">
        </a>
    </h4>
</div>

<strong>CR-Solver is a GPU-accelerated, optimization-based solver for tendon-driven continuum robots. It unifies inverse kinematics, trajectory planning, and path following within a single constrained nonlinear optimization framework, implemented in pure Python on JAX.</strong>

<!-- <div style="text-align:center">
<img src="assets/teaser.png"  width="100%" height="100%">
</div> -->

> Continuum robots provide intrinsic compliance, high dexterity, and safe physical interaction, yet most widely used planning libraries are grounded in rigid-body assumptions. CR-Solver bridges this gap by leveraging GPU-accelerated parallel optimization to deliver fast, accurate, and constraint-aware solutions for inverse kinematics, trajectory planning, and path following.

---

</div>

## 📢 News

* **[2026-06-30]** 🔥 CR-Solver is accepted to **IROS 2026**.
* **[2026-06-30]** 🚀 Code and evaluation scripts are open-sourced.

## 💡 Highlights

* **Configuration generality**. Seamlessly supports continuum robots with various scalable capabilities and numbers of segments, including extendable (variable-length) robots.
* **Robust parallel optimization**. A two-stage strategy — massively parallel seed sampling / coarse optimization followed by a GPU-accelerated gradient-based refinement — that unleashes GPU parallelism, improving robustness to initialization and reducing susceptibility to local minima.
* **Accessible tooling**. A concise and extensible codebase implemented in pure `Python` on [`JAX`](https://github.com/jax-ml/jax) (JIT + automatic differentiation, batched trust-region Levenberg–Marquardt), lowering the barrier to adoption and research.

## 🛠️ Usage

The Python package is named `soul`.

### Installation

We use [`uv`](https://github.com/astral-sh/uv) for environment management.

```bash
# CPU backend (macOS, or any machine without an NVIDIA GPU)
uv sync

# GPU backend (Linux + NVIDIA GPU + CUDA 12)
uv sync --extra cuda
```

> [!NOTE]
> GPU acceleration requires **Linux + NVIDIA GPU + CUDA 12**. On macOS, `uv sync` installs the CPU build of JAX, which is sufficient for functional testing but cannot reproduce the GPU benchmarks reported in the paper.

Alternatively, with `conda` + `pip`:

```bash
conda create -n cr_solver python=3.11 -y
conda activate cr_solver
pip install -r requirements.txt
```

### Examples

Interactive demos under `example/` and `demo/` use [`viser`](https://github.com/nerfstudio-project/viser) for browser-based visualization. After launching, open `http://localhost:8080` and drag the transform handles to solve in real time.

```bash
uv run python example/02_base_ik.py                     # base inverse kinematics
uv run python example/03_ik_with_coll.py                # collision-aware IK
uv run python example/05_motion_planning.py             # trajectory planning (trajopt / rrt / prm)
uv run python example/06_constraint_motion_planning.py  # path following
```

Minimal API for inverse kinematics:

```python
import jax
from soul.robots.cc_robot import CCRobot
from soul.solver import IKSolver

robot = CCRobot.from_config("configs/robots/cc.json")
solver = IKSolver(
    robot, num_seeds_init=10, num_seeds_final=1,
    total_steps=64, init_steps=6,
)
solve = jax.jit(solver.solve_ik_best)

cfg = solve(target_wxyz, target_position)  # quaternion (w, x, y, z) + xyz
pose = robot.forward_kinematics(cfg)
```

### Evaluation

Benchmarks are intended to run on a GPU (Linux). Example entry points:

```bash
# Inverse kinematics benchmark
uv run python benchmark/ik/ik_eval.py

# Motion planning benchmark (choose the number of segments)
uv run python benchmark/mp/mp_eval.py --section-num 4
```

## 📊 Results

All numbers below are measured on an NVIDIA RTX 4090 (24 GB). CPU baselines are run on dual Intel Xeon Platinum 8480+ processors.

### Inverse Kinematics

#### Standard Inverse Kinematics

| Method | Device | Success Rate (%) | Total Time (s) |
|--------|--------|-----------------|----------------|
| Derivative-free | CPU | 47.1 | 59.44 |
| Newton-Raphson | CPU | 94.2 | 10.54 |
| EMS | CPU | 100 | 1.59 |
| Ours | CPU | 100 | 404.289 |
| Ours* | GPU | 44.2 | 0.139 |
| **Ours** | **GPU** | **100** | **0.461** |

*\* indicates method without beam search.*

#### Collision-Aware Inverse Kinematics

| Method | Device | Segments | Success Rate (%) | Time (s) |
|--------|--------|----------|-----------------|-----------|
| **Non-extendable robots** |||||
| EMS | CPU | 3 | 100 | 3.05 |
| **Ours** | **GPU** | **3** | **100** | **0.428** |
| **Extendable robots** |||||
| CIDGIKc | CPU | 3 | 95 | 238.57 |
| | CPU | 4 | 99 | 305.09 |
| | CPU | 5 | 97 | 898.03 |
| | CPU | 6 | 97 | 2122.13 |
| **Ours** | **GPU** | **3** | **100** | **0.426** |
| | **GPU** | **4** | **100** | **0.559** |
| | **GPU** | **5** | **100** | **0.707** |
| | **GPU** | **6** | **100** | **0.900** |

### Motion Planning

Success across segment counts (robot length scaled per segment):

| Num Sections | Eval Num | Reachable (%) | Success (%) | Pos Error | Rot Error | Time (s) |
|--------------|----------|---------------|-------------|-----------|-----------|----------|
| 3 | 100 | 100.00 | 100.00 | 0.0003 | 0.0006 | 45.149 |
| 4 | 100 | 100.00 | 100.00 | 0.0000 | 0.0000 | 50.072 |
| 5 | 100 | 100.00 | 100.00 | 0.0000 | 0.0000 | 55.769 |
| 6 | 100 | 100.00 | 100.00 | 0.0000 | 0.0000 | 64.859 |

#### Baseline Comparison

| Method | Num Sections | Eval Num | Accuracy (%) | Success Rate Mean (%) | Position Error Mean (m) | Position Error Std (m) | Rotation Error Mean (rad) | Rotation Error Std (rad) | Time Mean (s) | Time Std (s) |
|-----------|--------------|----------|--------------|----------------------|-------------------------|------------------------|---------------------------|--------------------------|---------------|--------------|
| PRM       | 3            | 1000     | 67.2         | 81.0                 | 0.0443                  | 0.1331                 | 0.3830                    | 0.5613                   | 0.3691        | 0.2639       |
| PRM       | 4            | 1000     | 41.9         | 48.6                 | 0.0336                  | 0.1155                 | 0.2678                    | 0.4892                   | 0.7273        | 0.6726       |
| PRM       | 5            | 1000     | 28.2         | 29.1                 | 0.0066                  | 0.0261                 | 0.0868                    | 0.2863                   | 0.9188        | 0.7486       |
| PRM       | 6            | 1000     | 33.1         | 33.4                 | 0.0036                  | 0.0245                 | 0.0327                    | 0.1657                   | 0.8647        | 0.7498       |
| PRM+Opt   | 3            | 1000     | 67.2         | 81.0                 | 0.0449                  | 0.1334                 | 0.3830                    | 0.5613                   | 0.4497        | 0.3187       |
| PRM+Opt   | 4            | 1000     | 41.4         | 48.4                 | 0.0358                  | 0.1166                 | 0.2692                    | 0.4894                   | 0.8097        | 0.6815       |
| PRM+Opt   | 5            | 1000     | 27.4         | 28.5                 | 0.0092                  | 0.0271                 | 0.0934                    | 0.2998                   | 1.0199        | 0.7882       |
| PRM+Opt   | 6            | 1000     | 31.9         | 32.3                 | 0.0064                  | 0.0262                 | 0.0343                    | 0.1682                   | 0.9958        | 0.8387       |
| RRT       | 3            | 1000     | 64.0         | 83.1                 | 0.2152                  | 0.6676                 | 0.4821                    | 0.6833                   | 1.2418        | 3.4319       |
| RRT       | 4            | 1000     | 59.5         | 79.3                 | 0.3103                  | 0.8998                 | 0.4644                    | 0.7245                   | 1.7440        | 5.0057       |
| RRT       | 5            | 1000     | 62.7         | 74.2                 | 0.2644                  | 0.8876                 | 0.2627                    | 0.5954                   | 2.0365        | 6.3078       |
| RRT       | 6            | 1000     | 61.7         | 73.3                 | 0.3572                  | 1.0480                 | 0.2379                    | 0.5973                   | 2.6484        | 7.7196       |
| RRT+Opt   | 3            | 1000     | 64.0         | 83.1                 | 0.2155                  | 0.6658                 | 0.4801                    | 0.6781                   | 1.2976        | 3.4423       |
| RRT+Opt   | 4            | 1000     | 58.8         | 79.2                 | 0.3236                  | 0.9160                 | 0.4775                    | 0.7344                   | 1.8030        | 4.9464       |
| RRT+Opt   | 5            | 1000     | 62.2         | 74.1                 | 0.2763                  | 0.8999                 | 0.2718                    | 0.6029                   | 2.1947        | 6.6052       |
| RRT+Opt   | 6            | 1000     | 61.7         | 73.4                 | 0.3452                  | 1.0296                 | 0.2306                    | 0.5843                   | 2.8895        | 8.1198       |
| TRAJOPT   | 3            | 1000     | 79.6         | 100.0                | 0.0530                  | 0.1528                 | 0.4164                    | 0.5979                   | 0.1058        | 0.0783       |
| TRAJOPT   | 4            | 1000     | 83.8         | 100.0                | 0.0362                  | 0.1066                 | 0.3105                    | 0.5292                   | 0.1243        | 0.1236       |
| TRAJOPT   | 5            | 1000     | 93.3         | 100.0                | 0.0176                  | 0.0562                 | 0.1287                    | 0.3428                   | 0.1725        | 0.2498       |
| TRAJOPT   | 6            | 1000     | 95.2         | 100.0                | 0.0155                  | 0.0601                 | 0.0817                    | 0.3123                   | 0.2032        | 0.2254       |

#### Per-Stage Timing

| Num Sections | Eval Num | IK Time (ms)      | PRM Time (ms)      | Opt Time (ms)      | Total Time (ms)      |
|--------------|----------|-------------------|--------------------|--------------------|----------------------|
| 3            | 200      | 26.12 ± 1.16      | 358.75 ± 99.52     | 59.99 ± 17.44      | 444.86 ± 100.77      |

### Constraint Motion Planning (Path Following)

#### CR-Solver (Ours)

| Letter | Position Error (mean, mm) | Position Error (std, mm) | Rotation Error (mean, deg) | Rotation Error (std, deg) | Time (ms) |
|----------------|---------------------------|--------------------------|----------------------------|---------------------------|--------------------|
| Square         | 3.0258                    | 3.0793                   | 0.4731                     | 0.4298                    | 114.6061           |
| Sinusoidal     | 3.7589                    | 4.8344                   | 0.2475                     | 0.3489                    | 79.6404            |
| ICRA-Shaped    | 3.6259                    | 3.9024                   | 0.5730                     | 0.5872                    | 122.0320           |

#### Micsolver (Baseline)

| Letter | Position Error (mean, mm) | Position Error (std, mm) | Rotation Error (mean, deg) | Rotation Error (std, deg) | Time (ms) |
|----------------|---------------------------|--------------------------|----------------------------|---------------------------|------------------------|
| Square         | 54.1991                   | 33.8234                  | 0.4119                     | 0.0943                    | 440.3740               |
| Sinusoidal     | 168.0112                  | 96.8850                  | 0.0297                     | 0.2003                    | 559.0190               |
| ICRA-Shaped    | 60.7206                   | 33.5663                  | 0.3420                     | 0.1215                    | 444.6243               |

#### ICRA-Shaped Path, Per-Letter (Ours)

| Letter | Position Error (mean, mm) | Position Error (std, mm) | Rotation Error (mean, deg) | Rotation Error (std, deg) | Time (ms) |
|----------------|---------------------------|--------------------------|----------------------------|---------------------------|--------------------|
| I              | 1.1410                    | 1.1382                   | 0.0368                     | 0.0414                    | 134.3710           |
| C              | 3.4225                    | 3.4846                   | 0.9890                     | 1.0237                    | 114.5749           |
| R              | 2.2640                    | 2.4647                   | 0.7454                     | 0.5627                    | 124.5766           |
| A              | 1.5713                    | 3.2012                   | 0.5209                     | 0.7211                    | 114.6054           |

#### ICRA-Shaped Path, Per-Letter (Micsolver)

| Letter | Position Error (mean, mm) | Position Error (std, mm) | Rotation Error (mean, deg) | Rotation Error (std, deg) | Time (ms) |
|----------------|--------------------------|--------------------------|----------------------------|---------------------------|------------------------|
| I              | 35.8807                   | 26.2712                  | 0.3818                     | 0.0632                    | 435.4040               |
| C              | 57.7731                   | 29.2497                  | 0.3893                     | 0.1064                    | 435.0460               |
| R              | 75.3152                   | 37.0807                  | 0.2515                     | 0.1485                    | 457.7900               |
| A              | 73.9135                   | 41.6637                  | 0.3453                     | 0.1680                    | 450.2570               |

## 📝 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{yang2026crsolver,
  title={CR-Solver: GPU-Accelerated Kinematics Solver for Tendon-driven Continuum Robots},
  author={Yang, Heqing and Yi, Yang and Zhong, Linqing and Huang, Linjiang and Liu, Si},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```

## 📄 License

This project is licensed under the Apache-2.0 License. See [LICENSE](./LICENSE) for more information.

## 🙏 Acknowledgement

This project builds upon several excellent open-source efforts, including [jaxls](https://github.com/brentyi/jaxls), [JAX](https://github.com/jax-ml/jax), [viser](https://github.com/nerfstudio-project/viser), and [CoACD](https://github.com/SarahWeiii/CoACD). We thank the authors for releasing their code.
