#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export JAX_PLATFORMS=cuda
export PYTHONPATH=/home/yhq/workspace/yi_DiffSoft

PYTHON_BIN=/home/yhq/miniconda3/envs/diffsoft/bin/python
SCRIPT_PATH=/home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval.py

REPEAT_NUM=70

WORLD_CONFIGS=(
    "configs/maps/mp_scene/obstacles_13.pick_from_shelf.json"
    "configs/maps/mp_scene/mp_demo.json"
)

for iter in 1 3 5 20; do
    for config in "${WORLD_CONFIGS[@]}"; do
        for sec in 3 4 5 6; do
            $PYTHON_BIN $SCRIPT_PATH --section-num $sec --repeat-num $REPEAT_NUM --world-config $config --max-iter $iter --test-name mp_eval_max_iter_${iter}
        done
    done
done
