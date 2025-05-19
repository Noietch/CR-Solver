from __future__ import annotations

import json

import jax
import jax_dataclasses as jdc
import jaxls
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

# --- Dataclasses ---


@jdc.pytree_dataclass
class PCCModelConfig:
    """Configuration for the PCCModel."""

    num_sections: jdc.Static[int]
    num_points_per_section: jdc.Static[int]
    length: jdc.Static[float]

    # kappa_range and phi_range
    lower_limits_kappa: Float[Array, " n_act_sections"]
    upper_limits_kappa: Float[Array, " n_act_sections"]
    lower_limits_phi: Float[Array, " n_act_sections"]
    upper_limits_phi: Float[Array, " n_act_sections"]

    @classmethod
    def from_config(cls, config_dict: dict | str) -> PCCModelConfig:
        """Creates a config object from a dictionary."""
        if isinstance(config_dict, str):
            config_dict = json.load(open(config_dict))
        
        return cls(
            num_sections=config_dict["num_sections"],
            num_points_per_section=config_dict["num_points_per_section"],
            length=config_dict["length"],
            lower_limits_kappa=jnp.array(config_dict["lower_limits_kappa"]),
            upper_limits_kappa=jnp.array(config_dict["upper_limits_kappa"]),
            lower_limits_phi=jnp.array(config_dict["lower_limits_phi"]),
            upper_limits_phi=jnp.array(config_dict["upper_limits_phi"]),
        )


@jdc.pytree_dataclass
class ConstantCurvatureState:
    """
    State of the PCC model (kappa, phi per section).
    Length is fixed by PCCModelConfig.
    """

    kappa: Float[Array, "*batch num_sections"]  # Curvature for each section
    phi: Float[Array, "*batch num_sections"]  # Rotation angle (phi) for each section


# --- PCCModel Class ---
@jdc.pytree_dataclass
class PCCRobot:
    """A differentiable Piecewise Constant Curvature (PCC) kinematic model."""

    config: PCCModelConfig

    var_cls: jdc.Static[type[jaxls.Var[ConstantCurvatureState]]]

    @staticmethod
    def from_config(
        config_dict: dict,
        default_cfg: dict | None = None,
    ) -> PCCRobot:

        config = PCCModelConfig.from_config(config_dict)

        if default_cfg is None:
            default_cfg = ConstantCurvatureState(
                kappa=jnp.zeros((config.num_sections,)),
                phi=jnp.zeros((config.num_sections,)),
            )
        else:
            default_cfg = ConstantCurvatureState.from_dict(default_cfg)
        assert default_cfg.kappa.shape == (config.num_sections,)

        class StateVar(
            jaxls.Var[Array],
            default_factory=lambda: default_cfg,
        ): ...

        robot = PCCRobot(
            config=config,
            var_cls=StateVar,
        )

        return robot

    def _calculate_row(self, s, kappa, c_p, s_p):
        ks = kappa * s
        c_ks = jnp.cos(ks)
        s_ks = jnp.sin(ks)

        def compute_pos_kappa_zero(operands):
            s_op, _, _, _, _, _ = operands
            return jnp.array([0.0, 0.0, s_op])

        def compute_pos_kappa_nonzero(operands):
            _, cp_op, sp_op, cks_op, sks_op, k_op = operands
            inv_k = 1.0 / k_op
            px = inv_k * (-cp_op + cp_op * cks_op)
            py = inv_k * (-sp_op + sp_op * cks_op)
            pz = sks_op * inv_k
            return jnp.array([px, py, pz])

        branch_operands = (s, c_p, s_p, c_ks, s_ks, kappa)
        position_vector = jax.lax.cond(
            jnp.allclose(kappa, 0.0),
            compute_pos_kappa_zero,
            compute_pos_kappa_nonzero,
            branch_operands,
        )
        px, py, pz = position_vector[0], position_vector[1], position_vector[2]

        row_elements = jnp.array(
            [
                [c_p * c_ks, -s_p, -c_p * s_ks, px],
                [s_p * s_ks, c_p, -s_p * s_ks, py],
                [s_ks, 0.0, c_ks, pz],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return row_elements

    def compute_transform(self, i: int, state: ConstantCurvatureState):
        si = jnp.linspace(
            0.0, self.config.length, num=self.config.num_points_per_section
        )
        c_p = jnp.cos(state.phi[i])
        s_p = jnp.sin(state.phi[i])
        
        T_matrix_rows = jax.vmap(self._calculate_row, in_axes=(0, None, None, None))(
            si, state.kappa[i], c_p, s_p
        )

        return T_matrix_rows

    @jdc.jit
    def _forward_kinematics(self, state: ConstantCurvatureState) -> Float[Array, "num_sections*num_points_per_section 4 4"]:
        
        all_transforms = []
        previous_transform = jnp.eye(4)
        
        for i in range(self.config.num_sections):
            local_transform = self.compute_transform(i, state)
            prev_transform_expanded = jnp.expand_dims(previous_transform, axis=-3)
            transform = jnp.matmul(prev_transform_expanded, local_transform)
            all_transforms.append(transform)
            previous_transform = transform[..., -1, :, :]
        
        transform = jnp.concatenate(all_transforms, axis=-3)
        return transform


    def forward_kinematics(self, state: ConstantCurvatureState) -> Float[Array, "*batch num_sections*num_points_per_section 4 4"]:
        if state.kappa.ndim == 1:
            return self._forward_kinematics(state)
        else:
            return jax.vmap(self._forward_kinematics)(state)
