export const rl_4_policy_gradient = {
  slug: 'rl-4-policy-gradient',
  title: 'পলিসি গ্রেডিয়েন্ট: সরাসরি পলিসি অপ্টিমাইজেশন',
  description: 'Policy parameterization πθ, REINFORCE algorithm, policy gradient theorem, advantage function, variance reduction with baseline এবং PyTorch implementation — value-free RL-এর নতুন দিগন্ত বাংলায়।',
  date: 'মে ২০২৬',
  category: 'রিইনফোর্সমেন্ট লার্নিং',
  readTime: 15,
  content: `
    <h3>১. Value-Based vs Policy-Based RL</h3>
    <p>এখন পর্যন্ত আমরা <strong>value-based</strong> methods দেখেছি (Q-Learning, DQN) — এগুলো প্রথমে Q-function শেখে, তারপর সেটা থেকে policy derive করে।</p>
    <p><strong>Policy-Based methods</strong>-এ আমরা সরাসরি policy optimize করি — Q-function শেখার দরকার নেই।</p>
    <p>Policy-based methods কখন ভালো?</p>
    <ul>
      <li><strong>Continuous action spaces</strong>: Robot arm-এর joint angles কত হবে? argmax নেওয়া অসম্ভব infinite actions-এ</li>
      <li><strong>Stochastic policies</strong>: Poker-এ deterministic strategy সহজেই exploit করা যায়</li>
      <li><strong>High-dimensional action spaces</strong>: Text generation (vocabulary = 50,000+ tokens)</li>
      <li><strong>Better convergence properties</strong>: Value-based-এর oscillation সমস্যা নেই</li>
    </ul>

    <h3>২. Policy Parameterization</h3>
    <p>আমরা policy-কে neural network parameters θ দিয়ে parameterize করি:</p>
    <p><strong>π_θ(a | s) = P(A=a | S=s; θ)</strong></p>
    <p>লক্ষ্য: θ optimize করে expected total return maximize করা:</p>
    <p><strong>J(θ) = E_{τ~π_θ}[G(τ)] = E_{τ~π_θ}[Σ_t γ^t · r_t]</strong></p>
    <p>এখানে τ (trajectory) = একটি episode-এর সমস্ত (s, a, r) tuples।</p>
    <pre><code>import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    """
    Policy Network: state নিয়ে action probabilities দেয়
    Discrete actions-এর জন্য (softmax output)
    """
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        logits = self.network(x)
        # Softmax দিয়ে probability distribution
        return F.softmax(logits, dim=-1)

    def get_action(self, state):
        """State থেকে action sample করো"""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_t)

        # Action distribution থেকে sample
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class ContinuousPolicyNetwork(nn.Module):
    """
    Continuous action-এর জন্য Policy Network
    Gaussian distribution-এর mean এবং std output করে
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ContinuousPolicyNetwork, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        # Log std: learnable parameter
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        features = self.shared(x)
        mean = self.mean_layer(features)
        std = self.log_std.exp()
        return mean, std

    def get_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.forward(state_t)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.squeeze().numpy(), log_prob</code></pre>

    <h3>৩. Policy Gradient Theorem</h3>
    <p>J(θ)-এর gradient কীভাবে compute করব? সরাসরি differentiate করা কঠিন কারণ trajectory distribution policy-র উপর depend করে।</p>
    <p><strong>Policy Gradient Theorem</strong> এই সমস্যা সমাধান করে:</p>
    <p><strong>∇_θ J(θ) = E_{τ~π_θ}[Σ_t ∇_θ log π_θ(a_t | s_t) · G_t]</strong></p>
    <p>এই result-টি remarkable: আমরা environment-এর dynamics জানার দরকার নেই! শুধু log probability-র gradient এবং sampled return দরকার।</p>
    <p>Intuition: যদি কোনো action-এর পরে high return পাওয়া যায় (G_t বড়), তাহলে সেই action-এর probability বাড়াও (log π বাড়াও)। যদি low return পাওয়া যায়, probability কমাও।</p>

    <h3>৪. REINFORCE Algorithm</h3>
    <p><strong>REINFORCE</strong> (Williams, 1992) হলো সবচেয়ে simple policy gradient algorithm। এটি Monte Carlo sampling ব্যবহার করে:</p>
    <ol>
      <li>Current policy π_θ দিয়ে একটি complete episode collect করো</li>
      <li>প্রতিটি time step-এর জন্য return G_t compute করো</li>
      <li>Policy gradient দিয়ে θ update করো</li>
    </ol>
    <pre><code>class REINFORCE:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        # Episode-এর data store করার জন্য
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        action, log_prob = self.policy.get_action(state)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def compute_returns(self):
        """Discounted returns backward থেকে গণনা"""
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def update(self):
        """Policy gradient update"""
        returns = self.compute_returns()

        # Normalize returns (variance reduction)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        # L = -E[log π(a|s) * G_t]  (negative কারণ আমরা maximize করছি)
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Buffer clear করো
        self.log_probs = []
        self.rewards = []

        return loss.item()</code></pre>

    <h3>৫. REINFORCE Training Loop</h3>
    <pre><code>import gymnasium as gym

def train_reinforce(env_name='CartPole-v1', n_episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = REINFORCE(state_dim, n_actions, lr=1e-3, gamma=0.99)
    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        # Episode collect করো
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            total_reward += reward
            state = next_state

        # Episode শেষে policy update করো
        loss = agent.update()
        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1:4d} | Avg Reward: {avg:6.1f} | Loss: {loss:.4f}")

    env.close()
    return agent, episode_rewards

agent, rewards = train_reinforce(n_episodes=1000)
print(f"\nFinal 100-ep average: {np.mean(rewards[-100:]):.1f}")</code></pre>

    <h3>৬. High Variance সমস্যা এবং Baseline</h3>
    <p>REINFORCE-এর প্রধান সমস্যা হলো <strong>high variance</strong>। একই situation-এ ভিন্ন trajectories অনেক different returns দিতে পারে।</p>
    <p>সমাধান: Gradient estimate থেকে একটি <strong>baseline b(s)</strong> বাদ দাও:</p>
    <p><strong>∇_θ J(θ) = E[Σ_t ∇_θ log π_θ(a_t | s_t) · (G_t - b(s_t))]</strong></p>
    <p>Baseline বাদ দিলে gradient-এর <strong>expectation পরিবর্তন হয় না</strong> কিন্তু <strong>variance কমে</strong>।</p>
    <p>সবচেয়ে সাধারণ baseline হলো state value function V(s) — এটাই <strong>Advantage Function</strong>:</p>
    <p><strong>A(s_t, a_t) = G_t - V(s_t)</strong></p>
    <p>A(s, a) > 0 মানে এই action average-এর চেয়ে ভালো ছিল; A(s, a) &lt; 0 মানে খারাপ ছিল।</p>
    <pre><code>class REINFORCEWithBaseline:
    """REINFORCE with Value Function Baseline"""
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99):
        # Policy network
        self.policy = PolicyNetwork(state_dim, n_actions)

        # Value network (baseline)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=lr
        )
        self.gamma = gamma

        self.saved_log_probs = []
        self.saved_states = []
        self.rewards = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state)
        self.saved_states.append(state_t)

        action, log_prob = self.policy.get_action(state)
        self.saved_log_probs.append(log_prob)
        return action

    def update(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        states = torch.stack(self.saved_states)
        values = self.value_net(states).squeeze()

        # Advantage: return - baseline
        advantages = returns - values.detach()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss (REINFORCE with baseline)
        policy_loss = -torch.stack(self.saved_log_probs) * advantages

        # Value loss (MSE)
        value_loss = F.mse_loss(values, returns)

        # Update
        self.policy_optimizer.zero_grad()
        policy_loss.sum().backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Clear buffers
        self.saved_log_probs = []
        self.saved_states = []
        self.rewards = []

        return policy_loss.sum().item(), value_loss.item()</code></pre>

    <h3>৭. Policy Gradient Methods-এর সীমাবদ্ধতা</h3>
    <p>REINFORCE এবং policy gradient-এর সমস্যাসমূহ:</p>
    <ul>
      <li><strong>High variance</strong>: Monte Carlo return-এ noisy estimates</li>
      <li><strong>Sample inefficiency</strong>: পুরো episode দরকার প্রতিটি update-এর জন্য</li>
      <li><strong>Large updates</strong>: একটি bad update policy destroy করতে পারে (catastrophic forgetting)</li>
    </ul>
    <p>এই সমস্যাগুলো সমাধান করতেই পরের পর্বে <strong>Actor-Critic</strong> এবং <strong>PPO</strong> দেখব। Actor-Critic bootstrapped value estimates ব্যবহার করে variance কমায়, এবং PPO large updates prevent করে।</p>
  `
};
