
## Evaluation 

### Inverse Kinematics

```bash
CUDA_VISIBLE_DEVICES=3 python benchmark/ik_eval.py
```

| num sections | eval num | position error (×10⁻⁵) | rotation error (×10⁻⁵) | success rate (%)| soltion time(ms) |
|--------------|----------|------------------------|------------------------|------------------|-----------------|
| 2            | 1000     | 3.502                  | 1.583                  | 100.00           | 0.543           |
| 3            | 1000     | 1.582                  | 6.917                  | 99.20            | 0.678           |
| 4            | 1000     | 2.019                  | 4.816                  | 99.50            | 0.806           |
| 5            | 1000     | 2.503                  | 5.651                  | 99.90            | 0.912           |
| 6            | 1000     | 2.849                  | 6.277                  | 100.00           | 1.307           |