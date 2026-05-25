export const rl_2_qlearning = {
  slug: 'rl-2-qlearning',
  title: 'Q-Learning: মূল্য ফাংশন দিয়ে শেখা',
  description: 'Q-function, temporal difference learning, Q-learning update rule, epsilon-greedy exploration এবং Python grid world example — model-free RL-এর সবচেয়ে জনপ্রিয় algorithm বাংলায়।',
  date: 'মে ২০২৬',
  category: 'রিইনফোর্সমেন্ট লার্নিং',
  readTime: 15,
  content: `
    <h3>১. Model-Free RL কেন দরকার?</h3>
    <p>আগের পর্বে আমরা দেখেছি যে Bellman equation ব্যবহার করে Value Iteration দিয়ে optimal policy বের করা যায় — কিন্তু এর জন্য transition function P(s'|s,a) জানতে হয়। বাস্তব পৃথিবীতে এই model প্রায়ই অজানা থাকে।</p>
    <p><strong>Model-free RL</strong>-এ agent সরাসরি environment-এর সাথে interact করে শেখে — transition probabilities না জেনেই। Q-Learning হলো সবচেয়ে বিখ্যাত model-free RL algorithm।</p>

    <h3>২. Q-Function কী?</h3>
    <p><strong>Q-function (Action-Value Function)</strong> Q(s, a) হলো state s-এ action a নিলে এবং তারপর optimal policy follow করলে expected total discounted return:</p>
    <p><strong>Q*(s, a) = E[r + γ · max_{a'} Q*(s', a') | s, a]</strong></p>
    <p>Q-function-এর গুরুত্ব: যদি আমরা Q*(s, a) জানি, তাহলে optimal action সহজেই বের করা যায়:</p>
    <p><strong>π*(s) = argmax_a Q*(s, a)</strong></p>
    <p>অর্থাৎ, Q* জানলে model (transition function) দরকার নেই!</p>
    <pre><code>import numpy as np

# Q-table initialization (states x actions)
n_states = 16   # 4x4 GridWorld
n_actions = 4   # up, down, left, right

# সমস্ত Q-values শূন্য দিয়ে শুরু
Q = np.zeros((n_states, n_actions))
print(f"Q-table shape: {Q.shape}")
print(f"Initial Q-table (first 4 states):")
print(Q[:4])</code></pre>

    <h3>৩. Temporal Difference (TD) Learning</h3>
    <p><strong>Temporal Difference (TD) Learning</strong> হলো Q-Learning-এর মূল ধারণা। TD learning Monte Carlo এবং Dynamic Programming-এর সংমিশ্রণ:</p>
    <ul>
      <li><strong>Monte Carlo</strong>: পুরো episode শেষ হওয়ার পর update করে (high variance)</li>
      <li><strong>Dynamic Programming</strong>: model জানা দরকার (environment unknown)</li>
      <li><strong>TD Learning</strong>: প্রতিটি step-এর পর update করে, model ছাড়াই</li>
    </ul>
    <p><strong>TD Error (δ)</strong>:</p>
    <p><strong>δ = r + γ · V(s') - V(s)</strong></p>
    <p>এই error বলছে: আমরা কতটা "surprised" হয়েছি। Positive হলে আমাদের estimate কম ছিল, negative হলে বেশি ছিল।</p>

    <h3>৪. Q-Learning Update Rule</h3>
    <p>Q-Learning-এর update rule হলো TD learning-এর Q-function version:</p>
    <p><strong>Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') - Q(s, a)]</strong></p>
    <p>এখানে:</p>
    <ul>
      <li><strong>α</strong> (alpha) — learning rate (0 &lt; α ≤ 1): কত দ্রুত নতুন তথ্য গ্রহণ করবে</li>
      <li><strong>r + γ · max_{a'} Q(s', a')</strong> — TD target: আমাদের updated estimate</li>
      <li><strong>r + γ · max Q(s', a') - Q(s, a)</strong> — TD error: current estimate-এর সাথে পার্থক্য</li>
    </ul>
    <p>Q-Learning একটি <strong>off-policy</strong> algorithm — এটি একটি policy follow করে sample নেয় কিন্তু অন্য (greedy) policy-র জন্য Q-value update করে।</p>
    <pre><code># Q-Learning Update
def q_learning_update(Q, state, action, reward, next_state,
                       alpha=0.1, gamma=0.99):
    """
    Q-Learning update rule
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """
    # Current Q-value
    current_q = Q[state, action]

    # TD Target: best possible Q-value from next state
    if next_state is None:  # Terminal state
        td_target = reward
    else:
        td_target = reward + gamma * np.max(Q[next_state])

    # TD Error
    td_error = td_target - current_q

    # Update Q-value
    Q[state, action] = current_q + alpha * td_error

    return td_error

# উদাহরণ: একটি single update
Q = np.zeros((16, 4))
state, action = 0, 3      # state 0-এ right action
reward = -0.01
next_state = 1

td_error = q_learning_update(Q, state, action, reward, next_state)
print(f"Updated Q(0, right) = {Q[state, action]:.6f}")
print(f"TD Error = {td_error:.6f}")</code></pre>

    <h3>৫. Epsilon-Greedy Exploration</h3>
    <p>Q-Learning-এ একটি critical challenge হলো <strong>exploration vs exploitation trade-off</strong>:</p>
    <ul>
      <li><strong>Exploitation</strong>: সবচেয়ে ভালো জানা action নেওয়া (greedy)</li>
      <li><strong>Exploration</strong>: নতুন actions try করা (random)</li>
    </ul>
    <p>যদি শুধু exploit করি, তাহলে suboptimal policy-তে আটকে যাব। যদি শুধু explore করি, তাহলে কখনো শিখব না।</p>
    <p><strong>Epsilon-greedy policy</strong>:</p>
    <ul>
      <li>ε probability-তে random action (exploration)</li>
      <li>(1-ε) probability-তে greedy action (exploitation)</li>
    </ul>
    <p>Training-এর শুরুতে ε = 1.0 (pure exploration), ধীরে ধীরে ε decay করে 0.01-এ নামিয়ে আনা হয়।</p>
    <pre><code>class EpsilonGreedy:
    def __init__(self, n_actions, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, Q_values):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: best known action
            return np.argmax(Q_values)

    def decay(self):
        """Epsilon decay করা"""
        self.epsilon = max(self.epsilon_min,
                          self.epsilon * self.epsilon_decay)

# ব্যবহার
policy = EpsilonGreedy(n_actions=4)
Q_values = np.array([0.1, 0.5, 0.3, 0.2])

print(f"Initial epsilon: {policy.epsilon}")
for _ in range(1000):
    policy.decay()
print(f"After 1000 decays: {policy.epsilon:.4f}")</code></pre>

    <h3>৬. সম্পূর্ণ Q-Learning Implementation</h3>
    <pre><code>import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    """Simple 4x4 GridWorld Environment"""
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.actions = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
        self.goal = (size-1, size-1)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self._pos_to_state(self.pos)

    def _pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]

    def step(self, action):
        dr, dc = self.actions[action]
        new_r = max(0, min(self.size-1, self.pos[0] + dr))
        new_c = max(0, min(self.size-1, self.pos[1] + dc))
        self.pos = (new_r, new_c)
        state = self._pos_to_state(self.pos)

        if self.pos == self.goal:
            return state, 1.0, True
        return state, -0.01, False


def train_q_learning(env, n_episodes=500, alpha=0.1,
                     gamma=0.99, epsilon_start=1.0,
                     epsilon_min=0.01, epsilon_decay=0.995):
    """Q-Learning training loop"""
    Q = np.zeros((env.n_states, env.n_actions))
    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 200:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])

            # Environment step
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q-Learning update
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(Q[next_state])

            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            steps += 1

        # Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.3f}, "
                  f"Epsilon = {epsilon:.3f}")

    return Q, episode_rewards


# Training
env = GridWorld(size=4)
Q_optimal, rewards = train_q_learning(env, n_episodes=500)

print("\nOptimal Q-table (state 0):", Q_optimal[0])
print("Optimal action at state 0:", ['Up','Down','Left','Right'][np.argmax(Q_optimal[0])])</code></pre>

    <h3>৭. Q-Learning-এর Convergence</h3>
    <p>Q-Learning theoretically optimal Q*-এ converge করবে যদি:</p>
    <ul>
      <li>প্রতিটি (s, a) pair infinite বার visit করা হয়</li>
      <li>Learning rate α যথাযথভাবে decay করে: Σα = ∞ এবং Σα² &lt; ∞</li>
      <li>Rewards bounded (finite)</li>
    </ul>
    <p>Q-Learning-এর সীমাবদ্ধতা: states এবং actions-এর সংখ্যা বড় হলে Q-table অব্যবহারিক হয়ে পড়ে। উদাহরণস্বরূপ, Atari game-এ screen pixels = state হলে Q-table অসাধ্য বড় হয়। এই সমস্যার সমাধান পরের পর্বে — Deep Q-Network (DQN)।</p>

    <h3>৮. SARSA: On-Policy Alternative</h3>
    <p>Q-Learning-এর একটি on-policy variant হলো <strong>SARSA</strong>:</p>
    <p><strong>Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') - Q(s, a)]</strong></p>
    <p>পার্থক্য: SARSA পরবর্তী action a' actually নেওয়া action use করে (on-policy), কিন্তু Q-Learning max Q(s', a') use করে (off-policy)। SARSA বেশি conservative কিন্তু safer।</p>
    <pre><code>def sarsa_update(Q, state, action, reward, next_state,
                  next_action, alpha=0.1, gamma=0.99):
    """
    SARSA Update: on-policy TD control
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s', a') - Q(s,a)]
    Note: a' হলো next state-এ actually নেওয়া action
    """
    if next_state is None:
        td_target = reward
    else:
        td_target = reward + gamma * Q[next_state, next_action]

    Q[state, action] += alpha * (td_target - Q[state, action])
    return Q</code></pre>

    <p>Q-Learning এবং SARSA উভয়ই tabular RL-এর foundation। পরের পর্বে আমরা দেখব কীভাবে neural network ব্যবহার করে এই Q-function approximate করা যায় — যা DQN-এর মূল ধারণা।</p>
  `
};
