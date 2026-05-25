export const rl_5_actor_critic = {
  slug: 'rl-5-actor-critic',
  title: 'Actor-Critic ও PPO',
  description: 'Actor-Critic framework (policy + value network), A2C/A3C, Proximal Policy Optimization (PPO), clipping objective, কেন PPO stable — modern RL-এর সবচেয়ে জনপ্রিয় algorithms বাংলায়।',
  date: 'মে ২০২৬',
  category: 'রিইনফোর্সমেন্ট লার্নিং',
  readTime: 16,
  content: `
    <h3>১. Actor-Critic Framework</h3>
    <p>Actor-Critic হলো policy gradient এবং value-based methods-এর সংমিশ্রণ। দুটি neural network একসাথে কাজ করে:</p>
    <ul>
      <li><strong>Actor (π_θ)</strong>: Policy network — কোন action নেওয়া উচিত তা decide করে</li>
      <li><strong>Critic (V_φ)</strong>: Value network — current state কতটা ভালো তা evaluate করে</li>
    </ul>
    <p>মূল idea: Actor action নেয়, Critic তাকে বলে সেটা কতটা ভালো ছিল (advantage estimate)। Critic-এর feedback দিয়ে Actor improve করে।</p>
    <p><strong>REINFORCE vs Actor-Critic</strong>:</p>
    <ul>
      <li>REINFORCE: পুরো episode-এর Monte Carlo return G_t ব্যবহার করে (high variance)</li>
      <li>Actor-Critic: Bootstrapped TD estimate ব্যবহার করে (lower variance, some bias)</li>
    </ul>
    <pre><code>import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

class ActorCriticNetwork(nn.Module):
    """
    Shared backbone সহ Actor-Critic Network
    """
    def __init__(self, state_dim, n_actions, hidden_dim=256):
        super(ActorCriticNetwork, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head: action probabilities
        self.actor = nn.Linear(hidden_dim, n_actions)

        # Critic head: state value
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value

    def get_action_and_value(self, state):
        """Action sample করো এবং value estimate করো"""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.forward(state_t)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, value.squeeze(), entropy</code></pre>

    <h3>২. Advantage Function এবং TD Error</h3>
    <p>Actor-Critic-এ <strong>advantage function</strong> TD error দিয়ে approximate করা হয়:</p>
    <p><strong>A(s_t, a_t) ≈ δ_t = r_t + γ · V(s_{t+1}) - V(s_t)</strong></p>
    <p>এই TD error:</p>
    <ul>
      <li><strong>Positive</strong>: expected-এর চেয়ে বেশি reward পাওয়া গেছে — action ভালো ছিল</li>
      <li><strong>Negative</strong>: expected-এর চেয়ে কম reward — action খারাপ ছিল</li>
    </ul>
    <p><strong>N-step return</strong> দিয়ে আরো ভালো estimate পাওয়া যায়:</p>
    <p><strong>A_n(s_t, a_t) = r_t + γr_{t+1} + ... + γ^n V(s_{t+n}) - V(s_t)</strong></p>
    <pre><code>class A2CAgent:
    """Advantage Actor-Critic (A2C)"""
    def __init__(self, state_dim, n_actions,
                 lr=3e-4, gamma=0.99,
                 entropy_coef=0.01, value_coef=0.5,
                 n_steps=5):

        self.net = ActorCriticNetwork(state_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_steps = n_steps

        # Rollout buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        action, log_prob, value, entropy = \
            self.net.get_action_and_value(state)

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action

    def compute_gae(self, next_value, gamma=0.99, lam=0.95):
        """
        Generalized Advantage Estimation (GAE)
        High lambda: high variance, low bias (Monte Carlo-র দিকে)
        Low lambda: low variance, high bias (TD-র দিকে)
        """
        advantages = []
        gae = 0

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t]
                     + gamma * values[t+1] * (1 - self.dones[t])
                     - values[t])
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in
                   zip(advantages, self.values)]
        return advantages, returns

    def update(self, next_state, done):
        """A2C update"""
        with torch.no_grad():
            if done:
                next_value = torch.zeros(1)
            else:
                _, next_value = self.net(
                    torch.FloatTensor(next_state).unsqueeze(0)
                )
                next_value = next_value.squeeze()

        advantages, returns = self.compute_gae(next_value)

        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / \
                     (advantages.std() + 1e-8)

        # Policy loss: maximize expected advantage
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss: MSE between predicted and actual returns
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus (exploration উৎসাহিত করা)
        states_t = torch.FloatTensor(self.states)
        logits, _ = self.net(states_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().mean()

        # Total loss
        total_loss = (policy_loss
                      + self.value_coef * value_loss
                      - self.entropy_coef * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

        # Buffer clear
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

        return policy_loss.item(), value_loss.item()</code></pre>

    <h3>৩. Proximal Policy Optimization (PPO)</h3>
    <p>PPO হলো OpenAI-এর develop করা state-of-the-art RL algorithm (Schulman et al., 2017)। এটি Actor-Critic-এর উপরে built কিন্তু একটি critical problem সমাধান করে: <strong>large, destructive policy updates</strong>।</p>
    <p>সমস্যা: Policy gradient update অনেক বড় হলে policy সম্পূর্ণ destroy হয়ে যেতে পারে এবং recover করা কঠিন হয়।</p>
    <p>PPO-র সমাধান: <strong>Clipped Surrogate Objective</strong>:</p>
    <p><strong>L^CLIP(θ) = E_t[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]</strong></p>
    <p>যেখানে <strong>r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)</strong> হলো probability ratio।</p>
    <p>Clipping কীভাবে কাজ করে:</p>
    <ul>
      <li>r_t = 1: নতুন এবং পুরনো policy একই action-এ agree করে</li>
      <li>r_t > 1+ε: policy অনেক বেশি change হয়েছে — clip করো</li>
      <li>r_t &lt; 1-ε: policy অনেক বেশি কমেছে — clip করো</li>
    </ul>
    <pre><code>class PPOAgent:
    """Proximal Policy Optimization (PPO)"""
    def __init__(self, state_dim, n_actions,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_epsilon=0.2, entropy_coef=0.01,
                 value_coef=0.5, n_epochs=4,
                 batch_size=64, n_steps=2048):

        self.net = ActorCriticNetwork(state_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps

    def collect_rollout(self, env, n_steps):
        """n_steps-এর experience collect করো"""
        states, actions, log_probs, rewards = [], [], [], []
        values, dones = [], []

        state, _ = env.reset()

        for _ in range(n_steps):
            action, log_prob, value, _ = \
                self.net.get_action_and_value(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)

            state = next_state if not done else env.reset()[0]

        return (states, actions, log_probs, rewards, values, dones,
                state)

    def compute_gae(self, rewards, values, dones, last_state, last_done):
        """GAE advantage computation"""
        with torch.no_grad():
            if last_done:
                last_value = 0
            else:
                _, last_val = self.net(
                    torch.FloatTensor(last_state).unsqueeze(0)
                )
                last_value = last_val.item()

        advantages = np.zeros(len(rewards))
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = last_value
                next_done = last_done
            else:
                next_val = values[t+1]
                next_done = dones[t]

            delta = rewards[t] + self.gamma * next_val * (1-next_done) - values[t]
            gae = delta + self.gamma * self.lam * (1-next_done) * gae
            advantages[t] = gae

        returns = advantages + np.array(values)
        return advantages, returns

    def ppo_update(self, states, actions, old_log_probs,
                   advantages, returns):
        """PPO clipped objective দিয়ে multiple epochs update"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / \
                     (advantages.std() + 1e-8)

        n = len(states)
        total_policy_loss = 0
        total_value_loss = 0

        for epoch in range(self.n_epochs):
            # Random mini-batch indices
            indices = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start+self.batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # New log probabilities এবং values
                logits, values = self.net(batch_states)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Probability ratio r_t(theta)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio,
                                    1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        return total_policy_loss, total_value_loss</code></pre>

    <h3>৪. PPO Training Loop</h3>
    <pre><code>def train_ppo(env_name='CartPole-v1', total_steps=100000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=3e-4,
        n_steps=2048,
        n_epochs=4,
        batch_size=64,
        clip_epsilon=0.2
    )

    steps_done = 0
    update = 0
    all_rewards = []

    while steps_done < total_steps:
        # Rollout collect করো
        (states, actions, log_probs, rewards,
         values, dones, last_state) = agent.collect_rollout(
            env, agent.n_steps
        )

        steps_done += agent.n_steps

        # Advantage এবং returns গণনা করো
        advantages, returns = agent.compute_gae(
            rewards, values, dones, last_state,
            dones[-1]
        )

        # PPO update (multiple epochs)
        p_loss, v_loss = agent.ppo_update(
            states, actions, log_probs, advantages, returns
        )

        update += 1
        episode_reward = sum(rewards)
        all_rewards.append(episode_reward)

        if update % 5 == 0:
            print(f"Update {update:4d} | Steps: {steps_done:7d} | "
                  f"Reward: {np.mean(all_rewards[-5:]):6.1f} | "
                  f"PLoss: {p_loss:.4f} | VLoss: {v_loss:.4f}")

    env.close()
    return agent, all_rewards

agent, rewards = train_ppo('CartPole-v1', total_steps=100000)</code></pre>

    <h3>৫. Stable-Baselines3 দিয়ে PPO</h3>
    <pre><code># pip install stable-baselines3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Vectorized environments (parallel training)
vec_env = make_vec_env('CartPole-v1', n_envs=4)

# PPO model তৈরি
model = PPO(
    policy='MlpPolicy',
    env=vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1
)

# Training
model.learn(total_timesteps=100000)
model.save("ppo_cartpole")

# Evaluation
eval_env = gym.make('CartPole-v1')
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10
)
print(f"Mean reward: {mean_reward:.1f} +/- {std_reward:.1f}")</code></pre>

    <h3>৬. A2C vs A3C vs PPO</h3>
    <p>Modern Actor-Critic methods-এর তুলনা:</p>
    <ul>
      <li><strong>A2C</strong> (Advantage Actor-Critic): Synchronous, single-machine, simple এবং reproducible</li>
      <li><strong>A3C</strong> (Asynchronous A3C): Multiple parallel agents, asynchronous updates — faster কিন্তু less reproducible</li>
      <li><strong>PPO</strong>: Clipped objective দিয়ে stable, sample efficient, easy to tune — বর্তমানে সবচেয়ে popular</li>
    </ul>
    <p>কেন PPO এত জনপ্রিয়?</p>
    <ul>
      <li>Simple to implement এবং tune</li>
      <li>Stable training — large policy updates prevent করে</li>
      <li>Multiple epochs — একই data থেকে বেশি শেখা</li>
      <li>Broad applicability — robotics, games, NLP (RLHF) সব জায়গায় কাজ করে</li>
    </ul>
    <p>পরের পর্বে আমরা এই সব algorithm একসাথে ব্যবহার করে real projects তৈরি করব — CartPole master করা এবং Atari games খেলা।</p>
  `
};
