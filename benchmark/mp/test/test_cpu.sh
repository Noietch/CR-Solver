#!/bin/bash

export JAX_PLATFORMS=cpu
export PYTHONPATH=/home/yhq/workspace/yi_DiffSoft

PYTHON_BIN=/home/yhq/miniconda3/envs/diffsoft/bin/python
SCRIPT_PATH=/home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py

REPEAT_NUM=70

for sec in 3 4 5 6; do
    $PYTHON_BIN $SCRIPT_PATH --section-num $sec --repeat-num $REPEAT_NUM --world-config configs/maps/mp_scene/obstacles_random_section_${sec}.json
done

WORLD_CONFIGS=(
    "configs/maps/mp_scene/obstacles_13.pick_from_shelf.json"
    "configs/maps/mp_scene/obstacles_14.pick_from_bookshelf.json"
    "configs/maps/mp_scene/obstacles_15.grab_from_box.json"
    "configs/maps/mp_scene/mp_demo.json"
)

for config in "${WORLD_CONFIGS[@]}"; do
    for sec in 3 4 5 6; do
        $PYTHON_BIN $SCRIPT_PATH --section-num $sec --repeat-num $REPEAT_NUM --world-config $config
    done
done
