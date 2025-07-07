import jax
from typing import Sequence
from jax import Array
import jaxls

from soul.robots import PCCRobot, CCRobot
from soul.robots.pcc_robot import PCCState
from soul.geom import RobotCollision, CollGeom
from soul.costs import (
    self_collision_cost,
    tendon_similarity_cost,
    world_collision_cost,
    elastic_energy_cost,
)


class TendonFKSolver:
    def __init__(
        self,
        robot: PCCRobot | CCRobot,
        robot_coll: RobotCollision,
        total_steps: int,
    ):
        self.robot = robot
        self.coll = robot_coll
        self.total_steps = total_steps

    def pcc_fk(
        self,
        start_state: PCCState,
        world_coll_list: Sequence[CollGeom],
        tendon_target: Array,
    ):
        robot_var = self.robot.var_cls(0)

        factors = [
            elastic_energy_cost(robot_var, 0.05),
            # self_collision_cost(self.robot, self.coll, robot_var, 0.05, 100.0),
            tendon_similarity_cost(self.robot, robot_var, tendon_target, 100.0),
        ]
        factors.extend(
            [
                world_collision_cost(
                    self.robot, self.coll, robot_var, world_coll, 0.0, 10.0
                )
                for world_coll in world_coll_list
            ]
        )
        sol, summary = (
            jaxls.LeastSquaresProblem(factors, [robot_var])
            .analyze()
            .solve(
                initial_vals=jaxls.VarValues.make([robot_var.with_value(start_state)]),
                verbose=False,
                linear_solver="conjugate_gradient",
                return_summary=True,
            )
        )
        jax.debug.print("{}", summary)
        return sol[robot_var]
