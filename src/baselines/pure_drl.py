"""纯DRL基线 (B5)

实现真正的DQN训练：将FJSP-AGV调度建模为MDP，
用DQN学习从状态到调度规则的映射。
模拟C&OR 2025的方法。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from src.problem.instance import FJSPAGVInstance
from src.algorithm.nsga3.encoding import Chromosome, random_chromosome
from src.algorithm.nsga3.decoding import decode, evaluate
from src.algorithm.nsga3.selection import non_dominated_sort
from src.baselines.dispatching_rules import dispatching_rule_solve


# 16条复合调度规则（动作空间）
COMPOSITE_RULES = [
    ('SPT', 'SPT'), ('SPT', 'MIN_LOAD'), ('SPT', 'MIN_POWER'), ('SPT', 'RANDOM'),
    ('LPT', 'SPT'), ('LPT', 'MIN_LOAD'), ('LPT', 'MIN_POWER'), ('LPT', 'RANDOM'),
    ('MOPNR', 'SPT'), ('MOPNR', 'MIN_LOAD'), ('MOPNR', 'MIN_POWER'), ('MOPNR', 'RANDOM'),
    ('FIFO', 'SPT'), ('FIFO', 'MIN_LOAD'),
    ('RANDOM', 'MIN_LOAD'), ('RANDOM', 'RANDOM'),
]


class DQNNetwork(nn.Module):
    """DQN网络：从状态特征映射到各规则的Q值"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


def extract_state_features(instance: FJSPAGVInstance, schedule=None) -> np.ndarray:
    """提取12维状态特征（模拟C&OR 2025的状态空间）"""
    m = instance.num_machines
    k = instance.num_agv

    features = np.zeros(12, dtype=np.float32)

    if schedule is None:
        return features

    # 1-3: 机器利用率统计 (min, mean, max)
    workloads = np.zeros(m)
    for u in range(m):
        for (_, _, s, e) in schedule.machine_schedule.get(u, []):
            workloads[u] += e - s
    cmax = max(schedule.op_end.values()) if schedule.op_end else 1.0
    util = workloads / max(cmax, 1e-6)
    features[0] = util.min()
    features[1] = util.mean()
    features[2] = util.max()

    # 4-5: 机器队列长度统计
    queues = np.array([len(schedule.machine_schedule.get(u, [])) for u in range(m)])
    features[3] = queues.mean() / max(queues.max(), 1)
    features[4] = queues.std() / max(queues.max(), 1)

    # 6-7: AGV利用率
    agv_tasks = np.array([len(schedule.agv_schedule.get(l, [])) for l in range(k)])
    features[5] = agv_tasks.mean() / max(agv_tasks.max(), 1)
    features[6] = agv_tasks.std() / max(agv_tasks.max(), 1)

    # 8: 完成率
    total_ops = instance.total_operations
    done_ops = sum(1 for key in schedule.op_end if schedule.op_end[key] <= cmax)
    features[7] = done_ops / max(total_ops, 1)

    # 9: 归一化Cmax
    est_cmax = sum(max(instance.get_processing_time(i, j, u)
                       for u in instance.get_compatible_machines(i, j))
                   for i in range(instance.num_jobs)
                   for j in range(instance.num_operations[i])
                   if instance.get_compatible_machines(i, j))
    features[8] = cmax / max(est_cmax, 1)

    # 10: 负载不均衡度
    features[9] = (workloads.max() - workloads.min()) / max(cmax, 1e-6)

    # 11-12: 能耗相关
    if instance.machine_proc_power is not None:
        features[10] = (instance.machine_proc_power * workloads).sum() / max(cmax * m, 1)
        features[11] = instance.machine_idle_power.mean() / max(instance.machine_proc_power.max(), 1)

    return features


class PureDRL:
    """纯DRL基线（DQN训练+推理）

    训练阶段：在多个实例上训练DQN选择最优调度规则
    推理阶段：用训练好的DQN为每个实例选择规则并生成解
    """

    def __init__(self, instance: FJSPAGVInstance,
                 num_episodes: int = 100,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 seed: int = 42,
                 verbose: bool = True):
        self.instance = instance
        self.num_episodes = num_episodes
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.verbose = verbose
        self.history = {'hv': [], 'gen': []}

        self.state_dim = 12
        self.action_dim = len(COMPOSITE_RULES)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(5000)

    def _select_action(self, state: np.ndarray) -> int:
        """epsilon-greedy动作选择"""
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.action_dim)
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            return int(q.argmax(dim=1).item())

    def _update(self):
        """DQN网络更新"""
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 当前Q值
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # 目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self) -> list:
        """训练DQN并生成解集"""
        best_solutions = []

        # ===== 训练阶段 =====
        for episode in range(self.num_episodes):
            # 随机选一个规则作为初始方案
            init_action = self.rng.randint(0, self.action_dim)
            op_rule, mac_rule = COMPOSITE_RULES[init_action]
            chrom = dispatching_rule_solve(self.instance, op_rule, mac_rule,
                                           seed=self.rng.randint(0, 10000))

            state = extract_state_features(self.instance, chrom.schedule)
            total_reward = 0

            # 每个episode尝试多次选择不同规则来改进
            for step in range(5):
                action = self._select_action(state)
                op_rule, mac_rule = COMPOSITE_RULES[action]
                new_chrom = dispatching_rule_solve(self.instance, op_rule, mac_rule,
                                                   seed=self.rng.randint(0, 10000))

                next_state = extract_state_features(self.instance, new_chrom.schedule)

                # 奖励：Cmax的改善
                old_cmax = chrom.objectives.makespan
                new_cmax = new_chrom.objectives.makespan
                reward = (old_cmax - new_cmax) / max(old_cmax, 1e-6)

                done = (step == 4)
                self.replay.push(state, action, reward, next_state, done)

                self._update()

                state = next_state
                chrom = new_chrom
                total_reward += reward

                best_solutions.append(new_chrom)

            # 更新epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # 更新目标网络
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # ===== 推理阶段 =====
        # 用训练好的DQN选择最优规则，生成多组解
        self.epsilon = 0  # 纯贪心
        for _ in range(20):
            state = extract_state_features(self.instance)
            for step in range(3):
                action = self._select_action(state)
                op_rule, mac_rule = COMPOSITE_RULES[action]
                chrom = dispatching_rule_solve(self.instance, op_rule, mac_rule,
                                               seed=self.rng.randint(0, 10000))
                state = extract_state_features(self.instance, chrom.schedule)
                best_solutions.append(chrom)

        # 提取非支配解
        if not best_solutions:
            return []

        objs = np.array([c.objectives.to_array() for c in best_solutions])
        fronts = non_dominated_sort(objs)
        archive = [best_solutions[i] for i in fronts[0]]

        if self.verbose:
            print(f"PureDRL: {self.num_episodes} episodes trained, "
                  f"{len(best_solutions)} solutions -> {len(archive)} non-dominated")

        return archive[:100]
