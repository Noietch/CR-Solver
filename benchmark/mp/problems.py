import jax
import jaxlie
from jaxtyping import Array
import numpy as np
import jax.numpy as jnp
import os
from typing import Callable

from soul.robots.cc_robot import ConstantCurvatureState
from soul.robots.tdcr_robot import TDCRRobot
from soul.geom import RobotCollision, CollGeom
from benchmark.mp.utils import is_state_in_collision


def sample_states(robot: TDCRRobot, num_states: int) -> ConstantCurvatureState:
    random_key = jax.random.PRNGKey(42)
    random_key, subkey = jax.random.split(random_key)
    theta = jax.random.uniform(
        key=subkey,
        shape=(num_states, robot.config.num_sections),
        minval=robot.config.lower_limits_theta,
        maxval=robot.config.upper_limits_theta,
    )
    phi = jax.random.uniform(
        key=subkey,
        shape=(num_states, robot.config.num_sections),
        minval=robot.config.lower_limits_phi,
        maxval=robot.config.upper_limits_phi,
    )

    states = ConstantCurvatureState(
        base_position=jnp.zeros((num_states, 3)),
        theta=theta,
        phi=phi,
    )
    return states


class Problem:
    def __init__(
        self,
        sample_data_path: str,
        eval_num: int,
        robot: TDCRRobot,
        robot_coll: RobotCollision,
        world_geom: CollGeom,
        batched_fk: Callable,
        min_distance: float,
        start_from_initialization: bool,
    ):
        self.sample_data_path = sample_data_path
        self.eval_num = eval_num
        self.robot = robot
        self.robot_coll = robot_coll
        self.world_geom = world_geom
        self.batched_fk = batched_fk
        self.min_distance = min_distance
        self.start_from_initialization = start_from_initialization

    def sample_pool(self):
        is_collision_vmap = jax.vmap(
            is_state_in_collision, in_axes=(0, None, None, None)
        )
        pool_size = self.eval_num * 10  # Sample more to have options for pairing
        accepted_states = []
        total_sampled = 0
        max_pool_sampling_attempts = pool_size * 3  # Safety break for pool sampling

        print(f"Sampling a pool of {pool_size} collision-free states...")
        while (
            len(accepted_states) < pool_size
            and total_sampled < max_pool_sampling_attempts
        ):
            candidate_states = sample_states(self.robot, self.eval_num)
            total_sampled += self.eval_num

            collision_mask = is_collision_vmap(
                candidate_states, self.robot, self.robot_coll, self.world_geom
            )
            non_colliding_states = jax.tree_util.tree_map(
                lambda x: x[~collision_mask], candidate_states
            )

            num_non_colliding = non_colliding_states.theta.shape[0]
            if num_non_colliding > 0:
                current_pool_size = sum(s.theta.shape[0] for s in accepted_states)
                needed = pool_size - current_pool_size
                if num_non_colliding > needed:
                    non_colliding_states = jax.tree_util.tree_map(
                        lambda x: x[:needed], non_colliding_states
                    )

                accepted_states.append(non_colliding_states)
                new_pool_size = sum(s.theta.shape[0] for s in accepted_states)
                print(f"  ... pool has {new_pool_size}/{pool_size} states")
        state_pool = jax.tree_util.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *accepted_states
        )
        return state_pool

    def return_state(self, theta: Array, phi: Array) -> ConstantCurvatureState:
        return ConstantCurvatureState(
            base_position=jnp.zeros((theta.shape[0], 3)),
            theta=jnp.array(theta),
            phi=jnp.array(phi),
        )

    def gen_problems(self):
        print(
            f"Sampling {self.eval_num} collision-free start-end pairs with min distance {self.min_distance}..."
        )
        # Sample a large pool of collision-free states first.
        state_pool = self.sample_pool()

        fk_transforms_pool = self.batched_fk(state_pool)
        tip_transforms_pool = jaxlie.SE3.from_matrix(fk_transforms_pool[:, -1, ...])
        positions_pool = tip_transforms_pool.translation()

        # Pair states from the pool
        final_start_states_list = []
        final_end_states_list = []
        used_indices = jnp.zeros(state_pool.theta.shape[0], dtype=bool)
        num_available = state_pool.theta.shape[0]
        max_pairing_attempts = num_available * 10  # Safety break for pairing

        print("Pairing states...")

        if self.start_from_initialization:
            print("Adding initialization states (theta=0, phi=0) as start states...")
            # Create single initialization state with theta=0, phi=0
            init_state = self.return_state(
                theta=jnp.full((1, self.robot.config.num_sections), 1e-7),
                phi=jnp.full((1, self.robot.config.num_sections), 1e-7),
            )

            for _ in range(max_pairing_attempts):
                if len(final_start_states_list) >= self.eval_num:
                    break
                available_indices = jnp.where(~used_indices)[0]
                idx2 = np.random.choice(
                    np.array(available_indices), size=1, replace=False
                )
                idx2_int = int(idx2.item())

                init_fk_transforms = self.batched_fk(init_state)
                init_tip_transform = jaxlie.SE3.from_matrix(
                    init_fk_transforms[0, -1, ...]
                )

                pos1 = init_tip_transform.translation()
                pos2 = positions_pool[idx2_int]

                distance = jnp.linalg.norm(pos2 - pos1)

                if distance >= self.min_distance:
                    start_state = init_state
                    end_state = jax.tree_util.tree_map(
                        lambda x: x[idx2_int : idx2_int + 1], state_pool
                    )

                    final_start_states_list.append(start_state)
                    final_end_states_list.append(end_state)

                    used_indices = used_indices.at[idx2_int].set(True)

        else:
            for _ in range(max_pairing_attempts):
                if len(final_start_states_list) >= self.eval_num:
                    break
                # Get available indices
                available_indices = jnp.where(~used_indices)[0]
                # Pick two random, unique indices from the available pool
                idx1, idx2 = np.random.choice(
                    a=np.array(available_indices), size=2, replace=False
                )
                pos1 = positions_pool[idx1]
                pos2 = positions_pool[idx2]
                distance = jnp.linalg.norm(pos1 - pos2)

                if distance >= self.min_distance:
                    start_state = jax.tree_util.tree_map(
                        lambda x: x[idx1 : idx1 + 1], state_pool
                    )
                    end_state = jax.tree_util.tree_map(
                        lambda x: x[idx2 : idx2 + 1], state_pool
                    )

                    final_start_states_list.append(start_state)
                    final_end_states_list.append(end_state)

                    used_indices = used_indices.at[idx1].set(True)
                    used_indices = used_indices.at[idx2].set(True)

        start_states: ConstantCurvatureState = jax.tree_util.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *final_start_states_list
        )
        end_states: ConstantCurvatureState = jax.tree_util.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *final_end_states_list
        )

        self.save(start_states=start_states, end_states=end_states)
        print(f"Finished sampling {start_states.theta.shape[0]} start/end pairs.")
        return start_states, end_states

    def save(
        self,
        success_indices: Array = None,
        rename_suffix: str = "_filter",
        start_states: ConstantCurvatureState = None,
        end_states: ConstantCurvatureState = None,
    ):
        save_load_path = self.sample_data_path
        if success_indices is not None:
            start_states, end_states = self.load(
                save_load_path=save_load_path, success_indices=success_indices
            )
            print(
                f"Saving {len(success_indices)} successful trials back to {save_load_path}"
            )
            save_load_path = save_load_path.replace(".npz", f"{rename_suffix}.npz")
        print(f"Saving sampled states to {save_load_path}...")
        os.makedirs(os.path.dirname(save_load_path), exist_ok=True)
        np.savez(
            save_load_path,
            start_theta=start_states.theta,
            start_phi=start_states.phi,
            end_theta=end_states.theta,
            end_phi=end_states.phi,
        )

    def load(
        self,
        save_load_path: str,
        success_indices: Array = None,
    ) -> tuple[ConstantCurvatureState, ConstantCurvatureState]:
        if not (save_load_path and os.path.exists(save_load_path)):
            os.makedirs(os.path.dirname(save_load_path), exist_ok=True)
            start_states, end_states = self.gen_problems()
            return start_states, end_states

        print(f"Loading pre-sampled states from {save_load_path}...")
        data = np.load(save_load_path)

        # Load original states
        original_start_theta = data["start_theta"]
        original_start_phi = data["start_phi"]
        original_end_theta = data["end_theta"]
        original_end_phi = data["end_phi"]

        total_trials = original_start_theta.shape[0]
        print(f"Loaded {total_trials} start/end pairs.")

        if success_indices is not None and len(success_indices) > 0:
            print(f"Load {len(success_indices)} sucessful trials...")

            all_indices = jnp.arange(total_trials)
            success_mask = jnp.isin(all_indices, success_indices)

            filtered_start_theta = original_start_theta[success_mask]
            filtered_start_phi = original_start_phi[success_mask]
            filtered_end_theta = original_end_theta[success_mask]
            filtered_end_phi = original_end_phi[success_mask]

            start_states = self.return_state(
                theta=filtered_start_theta,
                phi=filtered_start_phi,
            )
            end_states = self.return_state(
                theta=filtered_end_theta,
                phi=filtered_end_phi,
            )
            print(
                f"Successfully removed failed trials. Remaining: {start_states.theta.shape[0]} pairs."
            )

        else:
            print("No failed indices provided, returning original states.")
            start_states = self.return_state(
                theta=original_start_theta,
                phi=original_start_phi,
            )
            end_states = self.return_state(
                theta=original_end_theta,
                phi=original_end_phi,
            )
        return start_states, end_states
