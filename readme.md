
## Evaluation 

### Inverse Kinematics

```bash
CUDA_VISIBLE_DEVICES=3 python benchmark/ik_eval.py
```

| num sections | eval num | position error (m) | rotation error (rad) | success rate (%) | total time (s) |
|--------------|----------|-------------------|-------------------|-----------------|----------------|
| 2 | 1000 | 3.52e-05 | 1.57e-04 | 100.00 | 0.540 |
| 3 | 1000 | 1.63e-04 | 6.84e-04 | 99.50 | 0.671 |
| 4 | 1000 | 2.00e-04 | 4.62e-04 | 99.70 | 0.805 |
| 5 | 1000 | 2.51e-04 | 5.64e-04 | 99.90 | 0.932 |
| 6 | 1000 | 2.77e-04 | 6.15e-04 | 100.00 | 1.300 |

### Inverse Kinematics with collision

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

### Inverse Kinematics with collision and extendable length

| Num Sections | Eval Num | Position Error(m) | Rotation Error(m) | Success Rate (%) | Total Time (s) |
|-------------|----------|-------------------|-------------------|------------------|----------------|
| 2 | 250 | 3.55e-04 | 1.32e-03 | 96.00 | 0.539 |
| 3 | 250 | 5.82e-04 | 1.29e-03 | 96.80 | 1.062 |
| 4 | 250 | 8.42e-04 | 1.50e-03 | 97.60 | 1.597 |
| 5 | 250 | 8.35e-04 | 8.57e-04 | 98.40 | 2.902 |
| 6 | 250 | 8.57e-04 | 7.75e-04 | 97.20 | 4.370 |

## TODOs
[ ]Benchmark IK with Collision && Motion Planning (Assigin To YY)

[ ](PRM*) Sampling Based Motion Planner [Try to do special for continuum robot] (Assigin To ZLQ)

[ ](Fatcor Graph | MPPI) Online Planning (Assigin To ZLQ)

[ ]Complex Environment Navigation (Assigin To YHQ)