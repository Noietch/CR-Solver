#!/bin/bash

export JAX_PLATFORMS=cpu
export PYTHONPATH=/home/yhq/workspace/yi_DiffSoft

/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 3 --repeat-num 50 --world-config configs/maps/mp_scene/obstacles_random_section_3.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 4 --repeat-num 50 --world-config configs/maps/mp_scene/obstacles_random_section_4.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 5 --repeat-num 70 --world-config configs/maps/mp_scene/obstacles_random_section_5.json

/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 3 --repeat-num 50 --world-config configs/maps/mp_scene/obstacles_13.pick_from_shelf.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 4 --repeat-num 50 --world-config configs/maps/mp_scene/obstacles_13.pick_from_shelf.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 5 --repeat-num 70 --world-config configs/maps/mp_scene/obstacles_13.pick_from_shelf.json

/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 3 --repeat-num 50 --world-config configs/maps/mp_scene/obstacles_14.pick_from_bookshelf.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 4 --repeat-num 50 --world-config configs/maps/mp_scene/obstacles_14.pick_from_bookshelf.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 5 --repeat-num 70 --world-config configs/maps/mp_scene/obstacles_14.pick_from_bookshelf.json

/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 3 --repeat-num 50 --world-config configs/maps/mp_scene/obstacles_14.pick_from_bookshelf.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 4 --repeat-num 50 --world-config configs/maps/mp_scene/obstacles_14.pick_from_bookshelf.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 5 --repeat-num 70 --world-config configs/maps/mp_scene/obstacles_14.pick_from_bookshelf.json

/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 3 --repeat-num 50 --world-config configs/maps/mp_scene/obstacles_15.grab_from_box.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 4 --repeat-num 60 --world-config configs/maps/mp_scene/obstacles_15.grab_from_box.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 5 --repeat-num 70 --world-config configs/maps/mp_scene/obstacles_15.grab_from_box.json

/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 3 --repeat-num 50 --world-config configs/maps/mp_scene/mp_demo.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 4 --repeat-num 60 --world-config configs/maps/mp_scene/mp_demo.json
/home/yhq/miniconda3/envs/diffsoft/bin/python /home/yhq/workspace/yi_DiffSoft/benchmark/mp/mp_eval_cpu.py --section-num 5 --repeat-num 70 --world-config configs/maps/mp_scene/mp_demo.json
