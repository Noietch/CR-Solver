
## Evaluation 

### Inverse Kinematics

```bash
CUDA_VISIBLE_DEVICES=3 python benchmark/ik_eval.py
```

| Num Sections | Eval Num | Position Error(m) | Rotation Error(m) | Success Rate (%) | Total Time (s) |
|-------------|----------|----------------|----------------|------------------|----------------|
| 2 | 1000 | 3.51e-05 | 1.59e-04 | 100.00 | 0.540 |
| 3 | 1000 | 1.59e-04 | 7.26e-04 | 99.10 | 0.670 |
| 4 | 1000 | 1.96e-04 | 4.70e-04 | 99.60 | 0.805 |
| 5 | 1000 | 2.50e-04 | 5.69e-04 | 99.70 | 0.931 |
| 6 | 1000 | 2.85e-04 | 6.16e-04 | 99.70 | 1.299 |

### Benchmark IK with Collision && Motion Planning

```bash
CUDA_VISIBLE_DEVICES=3 python benchmark/ik_eval_with_coll.py
```

| Num Sections | Eval Num | Position Error(m) | Rotation Error(m) | Success Rate (%) | Total Time (s) |
|-------------|----------|----------------|----------------|------------------|----------------|
| 2 | 250 | 7.11e-03 | 3.12e-02 | 99.60 | 0.445 |
| 3 | 250 | 3.10e-03 | 1.13e-02 | 100.00 | 0.607 |
| 4 | 250 | 1.11e-03 | 1.88e-03 | 100.00 | 0.912 |
| 5 | 250 | 1.68e-03 | 2.31e-03 | 100.00 | 1.346 |
| 6 | 250 | 2.10e-03 | 1.47e-03 | 100.00 | 1.941 |


## TODOs
[ ]Benchmark IK with Collision && Motion Planning (Assigin To YY)

[ ](PRM*) Sampling Based Motion Planner [Try to do special for continuum robot] (Assigin To ZLQ)

[ ](Fatcor Graph | MPPI) Online Planning (Assigin To ZLQ)

[ ]Complex Environment Navigation (Assigin To YHQ)