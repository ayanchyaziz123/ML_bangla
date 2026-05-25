export const rl_3_dqn = {
  slug: 'rl-3-dqn',
  title: 'DQN: ডিপ Q-নেটওয়ার্ক',
  description: 'Neural network দিয়ে Q-function approximate করা, experience replay, target network, DQN architecture এবং PyTorch দিয়ে সম্পূর্ণ DQN implementation — deep reinforcement learning-এর breakthrough বাংলায়।',
  date: 'মে ২০২৬',
  category: 'রিইনফোর্সমেন্ট লার্নিং',
  readTime: 16,
  content: `
    <h3>১. Tabular Q-Learning-এর সীমাবদ্ধতা</h3>
    <p>Q-Learning দারুণ algorithm, কিন্তু এটি একটি <strong>Q-table</strong> রাখে — প্রতিটি (state, action) pair-এর জন্য একটি value। বাস্তব সমস্যায় এটি অকার্যকর:</p>
    <ul>
      <li><strong>Atari Breakout</strong>: 84×84 grayscale screen, 4 frames stack = 84×84×4 = 28,224 dimensional state space → 10^67,000+ সম্ভাব্য states!</li>
      <li><strong>Robotics</strong>: continuous joint angles, velocities → infinite states</li>
      <li><strong>Chess</strong>: ~10^43 possible board positions</li>
    </ul>
    <p>সমাধান: Q-table-এর বদলে একটি <strong>neural network</strong> ব্যবহার করো যা Q-function approximate করবে:</p>
    <p><strong>Q(s, a; θ) ≈ Q*(s, a)</strong></p>
    <p>এই approach-ই হলো <strong>Deep Q-Network (DQN)</strong>, যা 2013-15 সালে DeepMind প্রথম publish করে।</p>

    <h3>২. Neural Network as Q-Function Approximator</h3>
    <p>DQN-এ neural network input হিসেবে state নেয় এবং output হিসেবে প্রতিটি action-এর Q-value দেয়:</p>
    <pre><code>import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network
    Input:  state (e.g., preprocessed game screen বা feature vector)
    Output: Q-values for each action
    """
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.network(x)


# CartPole-v1-এর জন্য DQN
# State: [cart_pos, cart_vel, pole_angle, pole_vel] = 4 dims
# Actions: [push_left, push_right] = 2
state_dim = 4
n_actions = 2

model = DQN(state_dim, n_actions)
print(model)

# Test forward pass
dummy_state = torch.FloatTensor([[0.1, -0.2, 0.05, 0.1]])
q_values = model(dummy_state)
print(f"Q-values: {q_values}")</code></pre>

    <h3>৩. Atari-র জন্য CNN DQN</h3>
    <p>Atari game-এ raw pixel input থেকে Q-values বের করতে CNN ব্যবহার করা হয়:</p>
    <pre><code>class AtariDQN(nn.Module):
    """
    DeepMind-এর original DQN architecture for Atari
    Input: 4 stacked 84x84 grayscale frames
    """
    def __init__(self, n_actions):
        super(AtariDQN, self).__init__()

        # Convolutional layers (feature extraction)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (4,84,84) -> (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (64,7,7)
            nn.ReLU()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

atari_dqn = AtariDQN(n_actions=18)  # Atari-তে max 18 actions
print(f"Atari DQN parameters: {sum(p.numel() for p in atari_dqn.parameters()):,}")</code></pre>

    <h3>৪. Experience Replay</h3>
    <p>Standard Q-Learning-এ একটি বড় সমস্যা আছে: consecutive samples highly correlated। যদি agent সামনে হাঁটছে, তাহলে পর পর সব samples similar — এতে neural network unstable হয়।</p>
    <p><strong>Experience Replay</strong> এই সমস্যা সমাধান করে:</p>
    <ol>
      <li>প্রতিটি step-এর experience (s, a, r, s', done) একটি <strong>replay buffer</strong>-এ store করো</li>
      <li>Training-এর সময় buffer থেকে <strong>random mini-batch</strong> sample করো</li>
      <li>এই random samples uncorrelated, তাই training stable</li>
    </ol>
    <pre><code>from collections import deque
import random

class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """একটি experience store করো"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Random mini-batch sample করো"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

# ব্যবহার
buffer = ReplayBuffer(capacity=10000)
print(f"Buffer capacity: {buffer.buffer.maxlen}")
print(f"Current size: {len(buffer)}")</code></pre>

    <h3>৫. Target Network</h3>
    <p>DQN-এর দ্বিতীয় গুরুত্বপূর্ণ innovation হলো <strong>Target Network</strong>। সমস্যা:</p>
    <p>Q-Learning loss: <strong>L = (r + γ · max Q(s', a'; θ) - Q(s, a; θ))²</strong></p>
    <p>এখানে target <strong>r + γ · max Q(s', a'; θ)</strong> এবং prediction <strong>Q(s, a; θ)</strong> উভয়ই একই network θ ব্যবহার করে। প্রতিটি update-এ target পরিবর্তন হয়, যা training unstable করে — "chasing a moving target"।</p>
    <p>সমাধান: একটি আলাদা <strong>target network θ⁻</strong> রাখো যা main network-এর পুরনো copy। এটি C steps পর পর update হয়:</p>
    <p><strong>L = (r + γ · max Q(s', a'; θ⁻) - Q(s, a; θ))²</strong></p>
    <pre><code>import copy

class DQNAgent:
    def __init__(self, state_dim, n_actions,
                 lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64,
                 target_update_freq=100):

        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        # Online network এবং Target network
        self.online_net = DQN(state_dim, n_actions)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.load_state_dict(self.online_net.state_dict())

        # Target network-এর parameters update হয় না directly
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=lr
        )
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.online_net(state_t)
            return q_values.argmax().item()

    def update_target_network(self):
        """Target network-কে online network-এর সাথে sync করা"""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self):
        """একটি training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Random mini-batch sample
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Current Q-values (online network)
        current_q = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Target Q-values (target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss: Mean Squared Error
        loss = F.mse_loss(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (stability-র জন্য)
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min,
                          self.epsilon * self.epsilon_decay)

        # Target network update
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()</code></pre>

    <h3>৬. DQN Training Loop</h3>
    <pre><code>import gymnasium as gym

def train_dqn(env_name='CartPole-v1', n_episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100
    )

    episode_rewards = []
    losses = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Action select করো
            action = agent.select_action(state)

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Experience store করো
            agent.replay_buffer.push(
                state, action, reward, next_state, float(done)
            )

            # Training step
            loss = agent.train_step()
            if loss:
                losses.append(loss)

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"Ep {episode+1:4d} | Avg Reward: {avg:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, episode_rewards

# Training চালানো
import numpy as np
agent, rewards = train_dqn('CartPole-v1', n_episodes=500)
print(f"\nFinal 50-episode average: {np.mean(rewards[-50:]):.1f}")</code></pre>

    <h3>৭. DQN-এর Key Innovations</h3>
    <p>DQN-এর তিনটি মূল innovation যা training stable করে:</p>
    <ul>
      <li><strong>Experience Replay</strong>: Correlation break করে, data reuse করে, sample efficiency বাড়ায়</li>
      <li><strong>Target Network</strong>: Training target stable রাখে, divergence prevent করে</li>
      <li><strong>Reward Clipping</strong>: Reward -1, 0, +1-এ clip করা (Atari-তে) — different games-এ consistent scale</li>
    </ul>
    <p>DQN-এর extensions:</p>
    <ul>
      <li><strong>Double DQN</strong>: Overestimation bias কমায় — action selection ও evaluation আলাদা network দিয়ে</li>
      <li><strong>Dueling DQN</strong>: V(s) এবং A(s,a) আলাদাভাবে estimate করে</li>
      <li><strong>Prioritized Experience Replay</strong>: Important transitions বেশি sample করে</li>
    </ul>
    <p>পরের পর্বে আমরা সম্পূর্ণ ভিন্ন approach দেখব — Policy Gradient, যেখানে Q-function-ই learn করার দরকার নেই।</p>
  `
};
