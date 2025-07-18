
# Evaluation Benchmark

## Inverse Kinematics 

### Performance Comparison

#### Standard Inverse Kinematics

| Method | Device | Success Rate (%) | Total Time (s) |
|--------|--------|-----------------|----------------|
| Derivative-free | CPU | 47.1 | 59.44 |
| Newton-Raphson | CPU | 94.2 | 10.54 |
| EMS | CPU | 100 | 1.59 |
| Ours | CPU | 100 | 404.289 |
| Ours* | GPU | 44.2 | 0.139 |
| **Ours** | **GPU** | **100** | **0.461** |

*\* indicates method without beam search*

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


## Motion Planning



## Constraint Motion Planning

| Letter | Position Error (mean, mm) | Position Error (std, mm) | Rotation Error (mean, deg) | Rotation Error (std, deg) |
|--------|---------------------------|--------------------------|----------------------------|---------------------------|
| I      | 4.9251                    | 4.6790                   | 2.1938                     | 1.9402                    |
| C      | 5.4997                    | 4.7640                   | 1.5702                     | 1.3810                    |
| R      | 12.6086                   | 13.1801                  | 2.6513                     | 2.8896                    |
| A      | 3.6013                    | 4.8701                   | 1.2159                     | 0.9629                    |