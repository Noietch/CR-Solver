
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

## TODOs
[ ] Benchmark IK with Collision

[ ] (PRM*) Sampling Based Motion Planner

[ ] （Fatcor Graph）Online Planning