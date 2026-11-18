"""Inferencia PPO independiente del código de entrenamiento y de SB3."""

import numpy as np
import torch
from torch import nn

from walker_lab.agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, model_dir, device="cpu"):
        super().__init__(model_dir, device)
        checkpoint = torch.load(self.model_dir / "policy.pt", map_location=self.device, weights_only=True)
        layers = []
        size = 24
        for width in checkpoint["hidden_sizes"]:
            layers.extend([nn.Linear(size, width), nn.Tanh()])
            size = width
        layers.append(nn.Linear(size, 4))
        self.policy = nn.Sequential(*layers).to(self.device)
        self.policy.load_state_dict(checkpoint["state_dict"])
        self.policy.eval()
        self.mean = checkpoint["obs_mean"].to(self.device)
        self.var = checkpoint["obs_var"].to(self.device)
        self.epsilon = checkpoint["epsilon"]
        self.clip_obs = checkpoint["clip_obs"]

    def reset(self):
        # La política MLP no tiene estado recurrente.
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (24,) or not np.isfinite(observation).all():
            raise ValueError("Se esperaba una observación finita con forma (24,).")
        with torch.inference_mode():
            obs = torch.as_tensor(observation, device=self.device)
            obs = ((obs - self.mean) / torch.sqrt(self.var + self.epsilon)).clamp(-self.clip_obs, self.clip_obs)
            return self.policy(obs).clamp(-1, 1).cpu().numpy()
