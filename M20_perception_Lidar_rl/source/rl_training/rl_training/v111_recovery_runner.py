"""Reference-anchored PPO without editing the installed RSL-RL implementation."""
import copy
import hashlib
import math
from itertools import chain
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from .v11_continuous_runner import V11ContinuousRunner


def reference_anchor_loss(predicted, target, observations, front_weight, weight):
    weights = torch.ones_like(predicted)
    x = observations
    ascent = ((x[:, 57] > .5) & (x[:, 58] > 0)) | ((x[:, 60] > .5) & (x[:, 61] > 0))
    for row in (1, 2, 4, 5):
        weights[:, row] = torch.where(ascent, front_weight, 1.)
    return weight*((predicted-target.detach()).square()*weights).mean()


class RecoveryPPO(PPO):
    def __init__(self, *args, teacher_checkpoint, replay_path, anchor_weight,
                 anchor_front_stair_weight, anchor_batch_size, **kwargs):
        super().__init__(*args, **kwargs)
        if not Path(teacher_checkpoint).is_file() or not Path(replay_path).is_file():
            raise ValueError('Recovery requires an explicit frozen teacher and deployment replay')
        if (not math.isfinite(anchor_weight) or anchor_weight <= 0
            or not 0 < anchor_front_stair_weight <= 1 or anchor_batch_size < 1):
            raise ValueError('Invalid reference anchor parameters')
        self.reference_sha256 = hashlib.sha256(Path(teacher_checkpoint).read_bytes()).hexdigest()
        self.replay_sha256 = hashlib.sha256(Path(replay_path).read_bytes()).hexdigest()
        checkpoint = torch.load(teacher_checkpoint, map_location=self.device, weights_only=True)
        self.teacher = copy.deepcopy(self.actor)
        self.teacher.load_state_dict(checkpoint['actor_state_dict'], strict=True)
        self.teacher.requires_grad_(False).eval()
        with np.load(replay_path) as replay:
            obs = torch.from_numpy(replay['observation'].copy()).float().to(self.device)
        if obs.ndim != 2 or obs.shape[1] != 63 or not len(obs) or not torch.isfinite(obs).all():
            raise ValueError('Recovery replay must contain finite native 63D observations')
        self.replay = obs
        self.anchor_weight = anchor_weight
        self.front_weight = anchor_front_stair_weight
        self.anchor_batch_size = anchor_batch_size
        # Keep the early representation; later features and decoder can learn
        # the new landing policy, with the frozen reference protecting drive.
        for name, parameter in self.actor.named_parameters():
            if name.startswith(('mlp.0.', 'mlp.2.', 'distribution.')):
                parameter.requires_grad_(False)
        self.anchor_losses = []
        self._anchor_hook = self.optimizer.register_step_pre_hook(self._anchor_step)

    def _anchor_step(self, optimizer, args, kwargs):
        indices = torch.randint(len(self.replay), (self.anchor_batch_size,), device=self.device)
        x = self.replay[indices]
        obs = TensorDict({'policy': x}, batch_size=[len(x)])
        with torch.no_grad():
            target = self.teacher(obs)
        with torch.enable_grad():
            predicted = self.actor(obs)
            loss = reference_anchor_loss(predicted, target, x, self.front_weight, self.anchor_weight)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in chain(self.actor.parameters(), self.critic.parameters())
            if p.requires_grad], self.max_grad_norm)
        self.anchor_losses.append(float(loss.detach()))

    def update(self):
        self.anchor_losses.clear()
        losses = super().update()
        losses['reference_anchor'] = sum(self.anchor_losses)/max(1, len(self.anchor_losses))
        return losses


class RecoveryRunner(V11ContinuousRunner):
    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos['m20_common_step_counter'] = int(self.env.unwrapped.common_step_counter)
        infos['reference_sha256'] = self.alg.reference_sha256
        infos['replay_sha256'] = self.alg.replay_sha256
        super().save(path, infos)
