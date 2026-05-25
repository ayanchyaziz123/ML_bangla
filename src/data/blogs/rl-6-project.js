export const rl_6_project = {
  slug: 'rl-6-project',
  title: 'RL প্রজেক্ট: CartPole ও Atari গেম',
  description: 'OpenAI Gymnasium দিয়ে DQN এজেন্ট তৈরি করুন এবং CartPole ও Atari পরিবেশে ট্রেন করুন',
  date: 'মে ২০২৫',
  category: 'রিইনফোর্সমেন্ট লার্নিং',
  readTime: 16,
  content: `
<h3>প্রজেক্ট পরিকল্পনা</h3>
<p>এই প্রজেক্টে আমরা দুটি পরিবেশে RL এজেন্ট ট্রেন করব:</p>
<ul>
<li><strong>CartPole-v1:</strong> সরল পরিবেশ — একটি খুঁটি ব্যালেন্স রাখা</li>
<li><strong>LunarLander-v2:</strong> আরও জটিল — চাঁদে নিরাপদ অবতরণ</li>
</ul>

<h3>পরিবেশ সেটআপ</h3>

<pre><code class="language-python">import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# পরিবেশ তৈরি ও পরীক্ষা
env = gym.make('CartPole-v1')
print(f"অবস্থা স্থান: {env.observation_space}")   # Box(4,)
print(f"অ্যাকশন স্থান: {env.action_space}")        # Discrete(2)

obs, info = env.reset()
print(f"প্রাথমিক অবস্থা: {obs}")
# [cart_pos, cart_vel, pole_angle, pole_angular_vel]
</code></pre>

<h3>DQN নেটওয়ার্ক আর্কিটেকচার</h3>

<pre><code class="language-python">class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)
</code></pre>

<h3>DQN এজেন্ট</h3>

<pre><code class="language-python">class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.batch_size = 64
        self.update_target_every = 100
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q_values = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            return q_values.argmax().item()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # বর্তমান Q মান
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # টার্গেট Q মান
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # এপসিলন হ্রাস
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # টার্গেট নেটওয়ার্ক আপডেট
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
</code></pre>

<h3>ট্রেনিং লুপ</h3>

<pre><code class="language-python">def train(env_name='CartPole-v1', n_episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    rewards_history = []
    avg_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)
        avg = np.mean(rewards_history[-100:])
        avg_rewards.append(avg)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}: Reward={total_reward:.0f}, "
                  f"Avg(100)={avg:.1f}, Epsilon={agent.epsilon:.3f}")

        # সমাধান: গড় পুরস্কার >= 475
        if avg >= 475:
            print(f"\\nসমাধান! Episode {episode+1}")
            break

    env.close()

    plt.figure(figsize=(10, 4))
    plt.plot(rewards_history, alpha=0.4, label='পুরস্কার')
    plt.plot(avg_rewards, label='গড় (১০০ এপিসোড)', linewidth=2)
    plt.axhline(y=475, color='r', linestyle='--', label='সমাধান সীমা')
    plt.xlabel('Episode')
    plt.ylabel('মোট পুরস্কার')
    plt.legend()
    plt.title(f'{env_name} DQN ট্রেনিং')
    plt.show()

    return agent

# CartPole ট্রেনিং
agent = train('CartPole-v1', n_episodes=500)
</code></pre>

<h3>stable-baselines3 দিয়ে LunarLander</h3>

<pre><code class="language-python">from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# পরিবেশ তৈরি
env = Monitor(gym.make('LunarLander-v2'))

# PPO মডেল — হাতে কোড না লিখে লাইব্রেরি ব্যবহার
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)

# ট্রেনিং
model.learn(total_timesteps=300_000)
model.save("lunar_ppo")

# মূল্যায়ন
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"গড় পুরস্কার: {mean_reward:.2f} +/- {std_reward:.2f}")
</code></pre>

<h3>মডেল মূল্যায়ন ও ভিজুয়ালাইজেশন</h3>

<pre><code class="language-python">def evaluate_agent(agent, env_name='CartPole-v1', n_episodes=10):
    env = gym.make(env_name, render_mode='rgb_array')
    total_rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            with torch.no_grad():
                q_values = agent.policy_net(torch.FloatTensor(state).unsqueeze(0))
                action = q_values.argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(total_reward)

    env.close()
    print(f"মূল্যায়ন ({n_episodes} এপিসোড):")
    print(f"গড় পুরস্কার: {np.mean(total_rewards):.1f}")
    print(f"সর্বোচ্চ: {np.max(total_rewards):.1f}")
    print(f"সর্বনিম্ন: {np.min(total_rewards):.1f}")

evaluate_agent(agent)
</code></pre>

<h4>ফলাফল তুলনা</h4>
<table>
<tr><th>পরিবেশ</th><th>অ্যালগরিদম</th><th>সমাধান এপিসোড</th><th>চূড়ান্ত পুরস্কার</th></tr>
<tr><td>CartPole-v1</td><td>DQN</td><td>~200</td><td>500 (সর্বোচ্চ)</td></tr>
<tr><td>LunarLander-v2</td><td>PPO</td><td>~1000</td><td>200+</td></tr>
</table>

<p>RL প্রজেক্ট সফলভাবে সম্পন্ন! পরবর্তী চ্যালেঞ্জ: Atari গেম Pong বা Breakout-এ DQN ট্রেন করুন — ছবি থেকে সরাসরি খেলা শেখা।</p>
`
};
