"""V13.5-Final From-Scratch Actor: MLP backbone + GRU front-leg residual.

Design philosophy:
- Early training: nearly pure MLP (easy to explore, like V13)
- GRU residual: very weak initially, only affects front-leg gait memory
- Gradual transition: GRU takes over phase coordination as training progresses
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import unpad_trajectories


class V135FromScratchActorCritic(ActorCritic):
    """MLP backbone + GRU front-leg residual for V13.5-Final from-scratch training.

    Architecture:
        Base path: 63D obs → MLP[512,256,128] → 16D base_action
        Memory path: 63D obs → Linear(32) → GRU(64) → Linear(6D) front_leg_residual

    Output:
        action = base_action + g_up * front_leg_residual

    where:
        g_up: gate for ascent only (from obs)
        front_leg_residual: ±0.20 via tanh, only affects FL/FR 6 joints
    """

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
        init_noise_std=1.0,
        gru_hidden_dim=64,
        gru_input_dim=32,
        residual_scale=0.20,
        **kwargs,
    ):
        if kwargs:
            print(f"V135FromScratchActorCritic unused kwargs: {kwargs}")

        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
        )

        # GRU memory path for front-leg gait
        self.gru_input_proj = nn.Linear(num_actor_obs, gru_input_dim)
        self.gru = nn.GRU(gru_input_dim, gru_hidden_dim, batch_first=True)
        self.gru_output = nn.Linear(gru_hidden_dim, 6)  # 6D for FL/FR joints

        # Initialize GRU output to near-zero so residual starts weak
        nn.init.normal_(self.gru_output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gru_output.bias)

        self.gru_hidden_dim = gru_hidden_dim
        self.residual_scale = residual_scale
        self.hidden_state = None

        # Front-leg joint indices (will be set by runner)
        self.front_leg_joint_indices = None

        # Non-ascent frame counter for GRU reset
        self.non_ascent_frames = None

    def reset(self, dones=None):
        """Reset GRU hidden state on episode done."""
        super().reset(dones)
        if self.hidden_state is None:
            return
        if dones is None:
            # Full reset
            self.hidden_state = None
            if self.non_ascent_frames is not None:
                self.non_ascent_frames.zero_()
        else:
            # Mask reset
            keep = (dones != 1).float()[None, :, None]
            self.hidden_state = self.hidden_state * keep.to(self.hidden_state.dtype)
            if self.non_ascent_frames is not None:
                self.non_ascent_frames[dones == 1] = 0

    def act(self, observations, masks=None, hidden_states=None):
        """Forward pass with GRU gait residual.

        Args:
            observations: [num_envs, num_obs] or [num_envs, time_steps, num_obs]
            masks: unused (for compatibility)
            hidden_states: [1, num_envs, hidden_dim] or None

        Returns:
            actions: [num_envs, num_actions] or [num_envs, time_steps, num_actions]
            hidden_states: [1, num_envs, hidden_dim]
        """
        # MLP backbone: base action
        base_action_params = self.actor(observations)
        base_action = base_action_params  # No std split here, done by runner

        # GRU memory path
        is_single_step = observations.ndim == 2
        if is_single_step:
            obs = observations.unsqueeze(1)  # [num_envs, 1, num_obs]
        else:
            obs = observations

        # Project to GRU input
        gru_input = self.gru_input_proj(obs)  # [num_envs, time, gru_input_dim]

        # GRU forward
        if hidden_states is None:
            hidden_states = self.hidden_state

        gru_out, new_hidden = self.gru(gru_input, hidden_states)

        # GRU output: 6D front-leg residual
        residual_raw = self.gru_output(gru_out)  # [num_envs, time, 6]
        residual = self.residual_scale * torch.tanh(residual_raw)

        if is_single_step:
            residual = residual.squeeze(1)  # [num_envs, 6]

        # Extract ascent gate from observations
        # Assume obs includes ascent_active flag (will be provided by env)
        # For now, use a placeholder - runner will need to inject this
        g_up = self._extract_ascent_gate(observations)

        # Apply residual only to front-leg joints
        action = base_action.clone()
        if self.front_leg_joint_indices is not None:
            action[..., self.front_leg_joint_indices] += g_up.unsqueeze(-1) * residual

        # Update hidden state
        self.hidden_state = new_hidden

        # GRU reset: if non-ascent for 3 consecutive frames, zero hidden
        self._update_gru_reset(g_up)

        return action, new_hidden

    def act_inference(self, observations):
        """Inference mode: single-step forward."""
        actions, _ = self.act(observations)
        return actions

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Critic forward (no GRU, pure MLP)."""
        return self.critic(critic_observations)

    def _extract_ascent_gate(self, observations):
        """Extract ascent gate from observations.

        This is a placeholder - the actual implementation should extract
        the ascent_active flag from observations.

        For now, assume it's the last dimension or will be injected by runner.
        """
        # TODO: Extract from observations properly
        # Expected: observations should include a binary ascent_active flag
        if observations.ndim == 2:
            # [num_envs, num_obs]
            num_envs = observations.shape[0]
            # Placeholder: assume all ascending for now
            return torch.ones(num_envs, device=observations.device)
        else:
            # [num_envs, time, num_obs]
            num_envs, time_steps = observations.shape[:2]
            return torch.ones(num_envs, time_steps, device=observations.device)

    def _update_gru_reset(self, g_up):
        """Update GRU reset counter and zero hidden if non-ascent for 3 frames.

        Args:
            g_up: [num_envs] or [num_envs, time] ascent gate
        """
        if g_up.ndim > 1:
            # Take last timestep for recurrent rollout
            g_up = g_up[:, -1]

        if self.non_ascent_frames is None:
            self.non_ascent_frames = torch.zeros(
                g_up.shape[0], dtype=torch.long, device=g_up.device
            )

        # Update counter
        self.non_ascent_frames = torch.where(
            g_up > 0.5,
            torch.zeros_like(self.non_ascent_frames),  # Reset counter if ascending
            self.non_ascent_frames + 1,  # Increment if not ascending
        )

        # Zero hidden if non-ascent for >= 3 frames
        if self.hidden_state is not None:
            reset_mask = (self.non_ascent_frames >= 3).float()[None, :, None]
            self.hidden_state = self.hidden_state * (1 - reset_mask)

    def get_hidden_states(self):
        """Return current hidden states for logging/checkpoint."""
        return self.hidden_state


__all__ = ["V135FromScratchActorCritic"]
