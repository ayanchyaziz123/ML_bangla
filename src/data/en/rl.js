export const rlEn = [
  {
    slug: 'rl-1-mdp',
    title: 'MDP: Foundation of Reinforcement Learning',
    description: 'Markov Decision Processes, Bellman equations, value functions, and policy definitions',
    category: 'Reinforcement Learning',
    content: `
<h3>The RL Framework</h3>
<p>In Reinforcement Learning, an <strong>agent</strong> interacts with an <strong>environment</strong> by taking actions and receiving rewards. The goal is to find a policy that maximizes cumulative discounted reward.</p>

<h4>Key Components</h4>
<ul>
<li><strong>State (S):</strong> The current situation of the environment</li>
<li><strong>Action (A):</strong> What the agent can do</li>
<li><strong>Reward (R):</strong> Feedback signal from the environment</li>
<li><strong>Policy (π):</strong> Strategy mapping states to actions</li>
<li><strong>Value Function V(s):</strong> Expected cumulative reward from state s</li>
</ul>

<pre><code class="language-python">import numpy as np

# Bellman Equation
# V(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) * V(s')]

# Grid World Example
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.goal = (size-1, size-1)
        self.state = (0, 0)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        r, c = self.state
        if action == 0: r = max(0, r-1)      # Up
        elif action == 1: r = min(self.size-1, r+1)  # Down
        elif action == 2: c = max(0, c-1)    # Left
        elif action == 3: c = min(self.size-1, c+1)  # Right

        self.state = (r, c)
        done = self.state == self.goal
        reward = 1.0 if done else -0.01
        return self.state, reward, done

env = GridWorld(4)
state = env.reset()
print(f"Initial state: {state}")
next_state, reward, done = env.step(3)  # Move right
print(f"After action: {next_state}, reward: {reward}")
</code></pre>
`
  },
  {
    slug: 'rl-2-qlearning',
    title: 'Q-Learning: Value-Based Reinforcement Learning',
    description: 'Q-function, temporal difference learning, epsilon-greedy exploration, and grid world implementation',
    category: 'Reinforcement Learning',
    content: `
<h3>Q-Function</h3>
<p>Q(s,a) represents the expected cumulative reward when taking action a in state s and following the optimal policy thereafter.</p>
<p><strong>Q-Learning Update:</strong> Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]</p>

<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr; self.gamma = gamma; self.epsilon = epsilon

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        current_q = self.Q[state, action]
        target_q = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.lr * (target_q - current_q)

# Train on FrozenLake
import gymnasium as gym
env = gym.make('FrozenLake-v1', is_slippery=False)
agent = QLearning(16, 4, lr=0.1, gamma=0.99, epsilon=0.1)

rewards = []
for episode in range(5000):
    state, _ = env.reset()
    total_reward = 0
    for _ in range(100):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.update(state, action, reward, next_state, terminated or truncated)
        state = next_state
        total_reward += reward
        if terminated or truncated: break
    rewards.append(total_reward)

print(f"Win rate (last 500): {np.mean(rewards[-500:]):.2%}")
</code></pre>
`
  },
  {
    slug: 'rl-3-dqn',
    title: 'Deep Q-Networks (DQN)',
    description: 'Neural network for Q-function, experience replay, target network, and CartPole training',
    category: 'Reinforcement Learning',
    content: `
<h3>Why Deep Q-Networks?</h3>
<p>Q-tables only work for small discrete state spaces. DQN uses a neural network to approximate Q(s,a) for continuous or large state spaces.</p>
<p>Two key innovations: <strong>Experience Replay</strong> (breaks correlation) and <strong>Target Network</strong> (stabilizes training).</p>

<pre><code class="language-python">import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import random
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args): self.buffer.append(args)
    def sample(self, n):
        batch = random.sample(self.buffer, n)
        return [torch.FloatTensor(np.array(x)) for x in zip(*batch)]
    def __len__(self): return len(self.buffer)

env = gym.make('CartPole-v1')
state_dim, action_dim = 4, 2

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()
epsilon = 1.0

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0
    while True:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state)).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, [action], [reward], next_state, [float(done)])
        state = next_state; total_reward += reward

        if len(buffer) >= 64:
            states, actions, rewards, next_states, dones = buffer.sample(64)
            actions = actions.long()
            curr_q = policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + 0.99 * next_q * (1 - dones)
            loss = nn.MSELoss()(curr_q, target_q)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if done: break

    epsilon = max(0.01, epsilon * 0.995)
    if (episode + 1) % 100 == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode+1}: Reward={total_reward:.0f}, eps={epsilon:.3f}")
</code></pre>
`
  },
  {
    slug: 'rl-4-policy-gradient',
    title: 'Policy Gradient Methods',
    description: 'REINFORCE algorithm, policy gradient theorem, baseline, and PyTorch implementation',
    category: 'Reinforcement Learning',
    content: `
<h3>Policy Gradient vs Value-Based</h3>
<p>Instead of learning a value function and deriving a policy from it, policy gradient methods directly optimize the policy parameters θ to maximize expected reward.</p>
<p><strong>Policy Gradient Theorem:</strong> ∇J(θ) = E[∇log π_θ(a|s) · Q(s,a)]</p>

<pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x): return self.net(x)

def reinforce(env_name='CartPole-v1', n_episodes=1000, gamma=0.99, lr=1e-3):
    env = gym.make(env_name)
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(n_episodes):
        state, _ = env.reset()
        log_probs, rewards = [], []

        while True:
            probs = policy(torch.FloatTensor(state))
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            if terminated or truncated: break

        # Compute returns with baseline
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        loss = -torch.stack(log_probs) * returns
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Reward={sum(rewards):.0f}")

    return policy

policy = reinforce()
</code></pre>
`
  },
  {
    slug: 'rl-5-actor-critic',
    title: 'Actor-Critic and PPO',
    description: 'Actor-Critic framework, A2C, Proximal Policy Optimization with stable-baselines3',
    category: 'Reinforcement Learning',
    content: `
<h3>Actor-Critic Architecture</h3>
<p>Combines value-based and policy-based methods: <strong>Actor</strong> selects actions (policy network), <strong>Critic</strong> evaluates them (value network). The critic's advantage estimate reduces variance in policy updates.</p>

<pre><code class="language-python">import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(128, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

# PPO Objective
# L_CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
# r_t = π_θ(a|s) / π_θ_old(a|s) — probability ratio

# With stable-baselines3 (recommended for production)
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make('LunarLander-v2')
model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=2048,
            batch_size=64, n_epochs=10, gamma=0.99, verbose=1)
model.learn(total_timesteps=500_000)
model.save("ppo_lunar")

from stable_baselines3.common.evaluation import evaluate_policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
</code></pre>
`
  },
  {
    slug: 'rl-6-project',
    title: 'RL Project: CartPole and Atari Games',
    description: 'Training DQN on CartPole and PPO on LunarLander with OpenAI Gymnasium',
    category: 'Reinforcement Learning',
    content: `
<h3>Project Overview</h3>
<p>We train two agents: DQN on CartPole-v1 (simple balance task) and PPO on LunarLander-v2 (complex continuous control), comparing their learning curves.</p>

<pre><code class="language-python">import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import numpy as np

# CartPole with DQN
cartpole_env = Monitor(gym.make('CartPole-v1'))
dqn_model = DQN('MlpPolicy', cartpole_env,
                learning_rate=1e-3, buffer_size=50000,
                learning_starts=1000, target_update_interval=500,
                exploration_fraction=0.2, verbose=0)

eval_callback = EvalCallback(Monitor(gym.make('CartPole-v1')),
                             eval_freq=1000, deterministic=True, render=False)
dqn_model.learn(total_timesteps=50000, callback=eval_callback)

mean_r, std_r = evaluate_policy(dqn_model, cartpole_env, n_eval_episodes=20)
print(f"DQN CartPole: {mean_r:.1f} +/- {std_r:.1f}")

# LunarLander with PPO
lunar_env = Monitor(gym.make('LunarLander-v2'))
ppo_model = PPO('MlpPolicy', lunar_env, learning_rate=3e-4, verbose=0)
ppo_model.learn(total_timesteps=300000)

mean_r, std_r = evaluate_policy(ppo_model, lunar_env, n_eval_episodes=10)
print(f"PPO LunarLander: {mean_r:.1f} +/- {std_r:.1f}")
# Score >= 200 is considered solved

# Atari: Pong (requires additional setup)
# pip install stable-baselines3[extra] ale-py
# from stable_baselines3.common.env_util import make_atari_env
# from stable_baselines3.common.vec_env import VecFrameStack
# env = VecFrameStack(make_atari_env('PongNoFrameskip-v4', n_envs=4), n_stack=4)
# model = PPO('CnnPolicy', env, verbose=1)
# model.learn(total_timesteps=10_000_000)
</code></pre>
`
  },
];
