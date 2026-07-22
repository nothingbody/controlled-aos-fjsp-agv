"""PPO智能体（用于GRL模块A和B）

对应论文第5章 5.5.2节和5.6.3节 式(79)-(93)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Optional


class ActorCritic(nn.Module):
    """Actor-Critic网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, state: torch.Tensor):
        logits = self.actor(state)
        value = self.critic(state)
        return logits, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """选择动作

        Returns:
            action, log_prob, value
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = probs.argmax(dim=-1)
        elif generator is not None:
            action = torch.multinomial(
                probs, num_samples=1, replacement=True, generator=generator
            ).squeeze(-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor):
        """评估已有动作

        Returns:
            log_prob, entropy, value
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """经验缓冲区"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.policy_versions = []

    def store(
        self, state, action, log_prob, reward, value, done=False,
        policy_version: int = 0,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.policy_versions.append(int(policy_version))

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.policy_versions.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO智能体

    对应论文式(83)-(86)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_range: float = 0.2,
                 n_epochs: int = 4, batch_size: int = 64,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 device: str = 'cpu', network_seed: Optional[int] = None,
                 action_seed: Optional[int] = None,
                 bc_seed: Optional[int] = None,
                 ppo_seed: Optional[int] = None):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden = 128
        self.lr = float(lr)
        self.policy_version = 0

        if network_seed is None:
            self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        else:
            devices = [] if self.device.type == "cpu" else [self.device.index or 0]
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(network_seed))
                self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.action_generator = None
        if action_seed is not None:
            self.action_generator = torch.Generator(device=self.device)
            self.action_generator.manual_seed(int(action_seed))
        self.bc_generator = None
        if bc_seed is not None:
            self.bc_generator = torch.Generator(device=self.device)
            self.bc_generator.manual_seed(int(bc_seed))
        self.ppo_rng = (
            np.random.RandomState(int(ppo_seed)) if ppo_seed is not None else None
        )

    def select(self, state: np.ndarray, deterministic: bool = False) -> int:
        """选择动作（推理模式）"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(
                state_t, deterministic, generator=self.action_generator
            )
        return action.item()

    def select_and_store(self, state: np.ndarray, reward: float = 0.0,
                         done: bool = False) -> int:
        """选择动作并存储经验"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(
                state_t, generator=self.action_generator
            )

        self.buffer.store(
            state=state,
            action=action.item(),
            log_prob=log_prob.item(),
            reward=reward,
            value=value.item(),
            done=done,
            policy_version=self.policy_version,
        )
        return action.item()

    def store_reward(self, reward: float, done: bool = False):
        """补充存储最后一个transition的奖励"""
        if len(self.buffer.rewards) > 0:
            self.buffer.rewards[-1] = reward
            self.buffer.dones[-1] = done

    def warm_start_behavior(self, states, actions, n_epochs: int = 10,
                            batch_size: int = 32, lr: float = 1e-3) -> dict:
        """Warm-start the actor by supervised imitation of external actions.

        This is deliberately separate from PPO.  In particular, demonstrations
        produced by UCB do not contain valid old-policy log probabilities and
        must never be inserted into the on-policy rollout buffer.
        """
        if len(states) < 2:
            return {}
        if len(states) != len(actions):
            raise ValueError("states and actions must have the same length")

        states_t = torch.as_tensor(
            np.asarray(states), dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(
            np.asarray(actions), dtype=torch.long, device=self.device
        )
        n = len(states_t)
        totals = {"bc_loss": 0.0, "bc_accuracy": 0.0, "bc_entropy": 0.0}
        updates = 0
        bc_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=lr)

        with torch.no_grad():
            pre_logits, _ = self.policy(states_t)
            pre_probs = F.softmax(pre_logits, dim=-1)
            totals["bc_pre_loss"] = float(
                F.cross_entropy(pre_logits, actions_t).item()
            )
            totals["bc_pre_accuracy"] = float(
                (pre_logits.argmax(dim=-1) == actions_t).float().mean().item()
            )
            totals["bc_pre_entropy"] = float(
                (-(pre_probs * torch.log(pre_probs.clamp_min(1e-12))).sum(dim=-1))
                .mean()
                .item()
            )

        epoch_loss = []
        epoch_accuracy = []
        epoch_entropy = []

        self.policy.train()
        for _ in range(n_epochs):
            for idx in torch.randperm(
                n, device=self.device, generator=self.bc_generator
            ).split(batch_size):
                logits, _ = self.policy(states_t[idx])
                loss = F.cross_entropy(logits, actions_t[idx])
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
                accuracy = (logits.argmax(dim=-1) == actions_t[idx]).float().mean()

                bc_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                bc_optimizer.step()

                totals["bc_loss"] += float(loss.item())
                totals["bc_accuracy"] += float(accuracy.item())
                totals["bc_entropy"] += float(entropy.item())
                updates += 1

            with torch.no_grad():
                full_logits, _ = self.policy(states_t)
                full_probs = F.softmax(full_logits, dim=-1)
                epoch_loss.append(float(F.cross_entropy(full_logits, actions_t).item()))
                epoch_accuracy.append(
                    float((full_logits.argmax(dim=-1) == actions_t).float().mean().item())
                )
                epoch_entropy.append(
                    float(
                        (-(full_probs * torch.log(full_probs.clamp_min(1e-12))).sum(dim=-1))
                        .mean()
                        .item()
                    )
                )

        if updates:
            for key in ("bc_loss", "bc_accuracy", "bc_entropy"):
                totals[key] /= updates

        with torch.no_grad():
            final_logits, _ = self.policy(states_t)
            final_probs = F.softmax(final_logits, dim=-1)
            totals["bc_final_loss"] = float(
                F.cross_entropy(final_logits, actions_t).item()
            )
            totals["bc_final_accuracy"] = float(
                (final_logits.argmax(dim=-1) == actions_t).float().mean().item()
            )
            totals["bc_final_entropy"] = float(
                (-(final_probs * torch.log(final_probs.clamp_min(1e-12))).sum(dim=-1))
                .mean()
                .item()
            )
            totals["bc_pre_post_kl"] = float(
                (
                    pre_probs
                    * (
                        torch.log(pre_probs.clamp_min(1e-12))
                        - torch.log(final_probs.clamp_min(1e-12))
                    )
                )
                .sum(dim=-1)
                .mean()
                .item()
            )
            predictions = final_logits.argmax(dim=-1)
            action_dim = int(final_logits.shape[-1])
            confusion = torch.zeros(
                (action_dim, action_dim), dtype=torch.int64, device=self.device
            )
            for true_action, predicted_action in zip(actions_t, predictions):
                confusion[true_action, predicted_action] += 1

        self.policy.eval()
        totals["bc_samples"] = int(n)
        totals["bc_updates"] = int(updates)
        totals["bc_epoch_loss"] = epoch_loss
        totals["bc_epoch_accuracy"] = epoch_accuracy
        totals["bc_epoch_entropy"] = epoch_entropy
        totals["bc_confusion_matrix"] = confusion.cpu().tolist()
        return totals

    def predict_value(self, state: np.ndarray) -> float:
        """Return the critic estimate used to bootstrap a truncated rollout."""
        with torch.no_grad():
            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            _, value = self.policy(state_t)
        return float(value.squeeze().item())

    def update(self, last_state: Optional[np.ndarray] = None) -> dict:
        """PPO策略更新

        Returns:
            训练统计信息
        """
        if len(self.buffer) < 2:
            self.buffer.clear()
            return {}

        behavior_versions = set(int(value) for value in self.buffer.policy_versions)
        if behavior_versions != {int(self.policy_version)}:
            raise RuntimeError(
                "PPO rollout mixes behavior-policy versions: "
                f"buffer={sorted(behavior_versions)} current={self.policy_version}"
            )
        behavior_policy_version = int(self.policy_version)

        # 计算GAE优势估计
        last_value = 0.0 if last_state is None else self.predict_value(last_state)
        advantages, returns = self._compute_gae(last_value=last_value)
        raw_advantages = advantages.copy()
        raw_returns = returns.copy()
        raw_rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        old_values_np = np.asarray(self.buffer.values, dtype=np.float32)

        # 转为tensor
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # 归一化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮更新
        stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'approx_kl': 0,
            'clip_fraction': 0,
            'gradient_norm': 0,
        }
        n = len(states)

        for _ in range(self.n_epochs):
            indices = (
                self.ppo_rng.permutation(n)
                if self.ppo_rng is not None
                else np.random.permutation(n)
            )
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                new_log_probs, entropy, new_values = self.policy.evaluate_action(
                    states[idx], actions[idx]
                )

                # 重要性采样比率
                ratio = torch.exp(new_log_probs - old_log_probs[idx])

                # Clipped目标
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_range,
                                    1 + self.clip_range) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(new_values, returns[idx])

                # 总损失
                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.mean().item()
                stats['approx_kl'] += (old_log_probs[idx] - new_log_probs).mean().item()
                stats['clip_fraction'] += (
                    (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()
                )
                stats['gradient_norm'] += float(gradient_norm.item())

        batches_per_epoch = max((n + self.batch_size - 1) // self.batch_size, 1)
        num_updates = max(self.n_epochs * batches_per_epoch, 1)
        for k in stats:
            stats[k] /= num_updates
        stats['rollout_size'] = int(n)
        stats['bootstrap_value'] = float(last_value)
        stats['optimizer_steps'] = int(num_updates)
        stats['advantage_mean_raw'] = float(np.mean(raw_advantages))
        stats['advantage_std_raw'] = float(np.std(raw_advantages))
        stats['return_mean'] = float(np.mean(raw_returns))
        stats['return_std'] = float(np.std(raw_returns))
        stats['reward_mean'] = float(np.mean(raw_rewards))
        stats['reward_std'] = float(np.std(raw_rewards))
        return_variance = float(np.var(raw_returns))
        stats['explained_variance_pre'] = (
            float(1.0 - np.var(raw_returns - old_values_np) / return_variance)
            if return_variance > 1e-12
            else 0.0
        )

        self.policy_version += 1
        stats['behavior_policy_version'] = behavior_policy_version
        stats['updated_policy_version'] = int(self.policy_version)
        self.buffer.clear()
        return stats

    def _compute_gae(self, last_value: float = 0.0):
        """计算广义优势估计 式(84)"""
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        n = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def save(self, path: str):
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'policy_version': int(self.policy_version),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.policy_version = int(checkpoint.get('policy_version', 0))
        self.buffer.clear()

    def export_training_state(self) -> dict:
        """Return a detached full PPO state for audited cross-episode training."""
        if len(self.buffer) != 0:
            raise RuntimeError("cannot export PPO state with a nonempty rollout buffer")
        return {
            "policy": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.policy.state_dict().items()
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "policy_version": int(self.policy_version),
            "architecture": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden": self.hidden,
            },
            "hyperparameters": {
                "lr": self.lr,
                "gamma": float(self.gamma),
                "gae_lambda": float(self.gae_lambda),
                "clip_range": float(self.clip_range),
                "n_epochs": int(self.n_epochs),
                "batch_size": int(self.batch_size),
                "entropy_coef": float(self.entropy_coef),
                "value_coef": float(self.value_coef),
            },
        }

    def load_training_state(self, state: dict, *, load_optimizer: bool) -> None:
        """Load parameters and optionally Adam moments; never load rollouts or RNGs."""
        architecture = dict(state.get("architecture", {}))
        expected = {"state_dim": self.state_dim, "action_dim": self.action_dim}
        observed = {key: int(architecture.get(key, -1)) for key in expected}
        if observed != expected:
            raise ValueError(
                f"incompatible PPO architecture: observed={observed} expected={expected}"
            )
        if int(architecture.get("hidden", -1)) != self.hidden:
            raise ValueError(
                f"incompatible PPO hidden width: {architecture.get('hidden')} "
                f"!= {self.hidden}"
            )
        hyperparameters = dict(state.get("hyperparameters", {}))
        expected_hyperparameters = {
            "lr": self.lr,
            "gamma": float(self.gamma),
            "gae_lambda": float(self.gae_lambda),
            "clip_range": float(self.clip_range),
            "n_epochs": int(self.n_epochs),
            "batch_size": int(self.batch_size),
            "entropy_coef": float(self.entropy_coef),
            "value_coef": float(self.value_coef),
        }
        mismatches = {
            key: {"observed": hyperparameters.get(key), "expected": value}
            for key, value in expected_hyperparameters.items()
            if hyperparameters.get(key) != value
        }
        if mismatches:
            raise ValueError(f"incompatible PPO hyperparameters: {mismatches}")
        self.policy.load_state_dict(state["policy"], strict=True)
        if load_optimizer:
            self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.policy_version = int(state.get("policy_version", 0))
        self.buffer.clear()
