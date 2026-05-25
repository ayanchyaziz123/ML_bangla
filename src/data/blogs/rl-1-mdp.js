export const rl_1_mdp = {
  slug: 'rl-1-mdp',
  title: 'MDP: রিইনফোর্সমেন্ট লার্নিং-এর ভিত্তি',
  description: 'Agent-environment interaction, Markov Decision Process, states, actions, rewards, transition function, discount factor, Bellman equation এবং value function — রিইনফোর্সমেন্ট লার্নিং-এর সম্পূর্ণ গাণিতিক ভিত্তি বাংলায়।',
  date: 'মে ২০২৬',
  category: 'রিইনফোর্সমেন্ট লার্নিং',
  readTime: 14,
  content: `
    <h3>১. রিইনফোর্সমেন্ট লার্নিং কী?</h3>
    <p>রিইনফোর্সমেন্ট লার্নিং (RL) হলো মেশিন লার্নিং-এর একটি শাখা যেখানে একটি <strong>agent</strong> তার <strong>environment</strong>-এর সাথে interaction করে শেখে। Supervised learning-এ labeled data থেকে শেখা হয়, কিন্তু RL-এ agent নিজে actions নেয় এবং সেই actions-এর ফলে পাওয়া <strong>reward</strong> থেকে শেখে।</p>
    <p>বাস্তব জীবনের উদাহরণ: একটি শিশু যখন হাঁটতে শেখে, সে পড়ে যায়, উঠে দাঁড়ায়, আবার চেষ্টা করে। পড়ে গেলে "negative reward" এবং সামনে এগোলে "positive reward" — এই feedback থেকেই সে হাঁটা শেখে। RL ঠিক এভাবেই কাজ করে।</p>
    <p>RL-এর প্রধান applications:</p>
    <ul>
      <li><strong>Game playing</strong> — AlphaGo, AlphaZero, OpenAI Five</li>
      <li><strong>Robotics</strong> — robot arm control, locomotion</li>
      <li><strong>Recommendation systems</strong> — personalized content</li>
      <li><strong>Autonomous driving</strong> — path planning</li>
      <li><strong>Finance</strong> — algorithmic trading</li>
    </ul>

    <h3>২. Agent-Environment Interaction</h3>
    <p>RL-এর মূল framework হলো agent এবং environment-এর মধ্যে cyclic interaction:</p>
    <ol>
      <li>Agent বর্তমান <strong>state s</strong> observe করে</li>
      <li>Agent একটি <strong>action a</strong> নেয়</li>
      <li>Environment নতুন <strong>state s'</strong> এবং <strong>reward r</strong> return করে</li>
      <li>প্রক্রিয়াটি পুনরাবৃত্তি হয়</li>
    </ol>
    <pre><code># RL Loop-এর basic structure
import gymnasium as gym

env = gym.make('CartPole-v1')
state, info = env.reset()

total_reward = 0
done = False

while not done:
    # Random policy (আপাতত)
    action = env.action_space.sample()

    # Environment-এ action নেওয়া
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    total_reward += reward
    state = next_state

print(f"Total reward: {total_reward}")
env.close()</code></pre>

    <h3>৩. Markov Decision Process (MDP)</h3>
    <p>RL-এর formal framework হলো <strong>Markov Decision Process (MDP)</strong>। একটি MDP পাঁচটি উপাদান দিয়ে define করা হয়:</p>
    <p><strong>MDP = (S, A, P, R, γ)</strong></p>
    <ul>
      <li><strong>S</strong> — State space: সমস্ত সম্ভাব্য states-এর set</li>
      <li><strong>A</strong> — Action space: সমস্ত সম্ভাব্য actions-এর set</li>
      <li><strong>P</strong> — Transition function: P(s' | s, a) — state s-এ action a নিলে state s'-এ যাওয়ার probability</li>
      <li><strong>R</strong> — Reward function: R(s, a, s') — state s থেকে action a নিয়ে s'-এ গেলে পাওয়া reward</li>
      <li><strong>γ</strong> — Discount factor: ভবিষ্যতের reward-এর বর্তমান মূল্য (0 ≤ γ ≤ 1)</li>
    </ul>

    <h4>Markov Property</h4>
    <p>MDP-এর সবচেয়ে গুরুত্বপূর্ণ assumption হলো <strong>Markov Property</strong>: ভবিষ্যত শুধু বর্তমান state-এর উপর নির্ভর করে, অতীতের পুরো history-র উপর নয়।</p>
    <p><strong>P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)</strong></p>
    <p>এই property-র কারণে আমাদের পুরো history মনে রাখতে হয় না — শুধু current state-ই যথেষ্ট।</p>
    <pre><code># Simple GridWorld MDP
import numpy as np

class GridWorldMDP:
    def __init__(self, size=4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, down, left, right
        self.actions = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
        self.goal_state = (size-1, size-1)
        self.reset()

    def state_to_pos(self, state):
        return (state // self.size, state % self.size)

    def pos_to_state(self, row, col):
        return row * self.size + col

    def reset(self):
        self.pos = (0, 0)
        return self.pos_to_state(*self.pos)

    def step(self, action):
        dr, dc = self.actions[action]
        new_r = max(0, min(self.size-1, self.pos[0] + dr))
        new_c = max(0, min(self.size-1, self.pos[1] + dc))
        self.pos = (new_r, new_c)

        state = self.pos_to_state(*self.pos)

        # Reward: goal-এ পৌঁছালে +1, অন্যথায় -0.01
        if self.pos == self.goal_state:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False

        return state, reward, done

env = GridWorldMDP(size=4)
state = env.reset()
print(f"Initial state: {state}, Position: {env.pos}")</code></pre>

    <h3>৪. Policy (পলিসি) π</h3>
    <p>একটি <strong>policy π</strong> হলো agent-এর behavior — কোন state-এ কোন action নেওয়া হবে তার mapping:</p>
    <ul>
      <li><strong>Deterministic policy:</strong> π(s) = a — state s-এ সবসময় action a নেওয়া হয়</li>
      <li><strong>Stochastic policy:</strong> π(a | s) = P(A=a | S=s) — state s-এ action a নেওয়ার probability</li>
    </ul>
    <p>আমাদের লক্ষ্য হলো এমন একটি <strong>optimal policy π*</strong> খুঁজে বের করা যা total cumulative reward maximize করে।</p>

    <h3>৫. Discount Factor γ এবং Return</h3>
    <p><strong>Discount factor γ</strong> (gamma) নির্ধারণ করে ভবিষ্যতের reward কতটা গুরুত্বপূর্ণ:</p>
    <ul>
      <li><strong>γ = 0</strong>: শুধু immediate reward গুরুত্বপূর্ণ (myopic)</li>
      <li><strong>γ = 1</strong>: সব future reward সমান গুরুত্বপূর্ণ</li>
      <li><strong>0 &lt; γ &lt; 1</strong>: কাছের reward বেশি গুরুত্বপূর্ণ (practical)</li>
    </ul>
    <p>Time step t থেকে শুরু করে total <strong>discounted return G_t</strong>:</p>
    <p><strong>G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... = Σ_{k=0}^{∞} γᵏ · r_{t+k}</strong></p>
    <pre><code># Discounted return গণনা
def compute_return(rewards, gamma=0.99):
    """
    rewards: list of rewards পাওয়া একটি episode-এ
    gamma: discount factor
    return: প্রতিটি timestep-এর discounted return
    """
    G = 0
    returns = []

    # পিছন থেকে গণনা করা (efficient)
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return returns

# উদাহরণ
rewards = [0, 0, 0, 1.0]  # শেষে goal পাওয়া
gamma = 0.9

returns = compute_return(rewards, gamma)
print("Rewards:", rewards)
print("Returns:", [round(g, 4) for g in returns])
# Returns: [0.729, 0.81, 0.9, 1.0]</code></pre>

    <h3>৬. Value Function V(s)</h3>
    <p><strong>State value function V^π(s)</strong> হলো policy π follow করলে state s থেকে expected total discounted return:</p>
    <p><strong>V^π(s) = E_π[G_t | S_t = s] = E_π[Σ_{k=0}^{∞} γᵏ · R_{t+k+1} | S_t = s]</strong></p>
    <p>এছাড়া <strong>Action-value function Q^π(s,a)</strong>:</p>
    <p><strong>Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]</strong></p>
    <p>V(s) এবং Q(s,a)-এর মধ্যে সম্পর্ক:</p>
    <p><strong>V^π(s) = Σ_a π(a|s) · Q^π(s,a)</strong></p>

    <h3>৭. Bellman Equation</h3>
    <p><strong>Bellman Equation</strong> হলো value function-এর recursive definition — এটি RL-এর সবচেয়ে গুরুত্বপূর্ণ equation:</p>
    <p><strong>V^π(s) = Σ_a π(a|s) · Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V^π(s')]</strong></p>
    <p>এই equation বলছে: state s-এর value = সম্ভাব্য actions এবং transitions-এর উপর weighted average of (immediate reward + discounted future value)।</p>
    <p><strong>Bellman Optimality Equation</strong> (optimal policy-র জন্য):</p>
    <p><strong>V*(s) = max_a Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V*(s')]</strong></p>
    <pre><code># Dynamic Programming দিয়ে Bellman equation solve করা
def value_iteration(env, gamma=0.99, theta=1e-6):
    """
    Value Iteration Algorithm
    - env: MDP environment
    - gamma: discount factor
    - theta: convergence threshold
    """
    n_states = env.n_states
    n_actions = env.n_actions
    V = np.zeros(n_states)

    iteration = 0
    while True:
        delta = 0
        for s in range(n_states):
            old_v = V[s]

            # সব actions-এর জন্য Q(s,a) গণনা
            action_values = []
            for a in range(n_actions):
                # Transition probabilities (deterministic env-এ simple)
                q_val = env.get_expected_return(s, a, V, gamma)
                action_values.append(q_val)

            # Bellman optimality: max action নেওয়া
            V[s] = max(action_values)
            delta = max(delta, abs(old_v - V[s]))

        iteration += 1
        if delta < theta:
            print(f"Converged after {iteration} iterations")
            break

    # Optimal policy extract করা
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = [env.get_expected_return(s, a, V, gamma)
                        for a in range(n_actions)]
        policy[s] = np.argmax(action_values)

    return V, policy

print("Value Iteration দিয়ে optimal value function এবং policy পাওয়া সম্ভব।")</code></pre>

    <h3>৮. MDP-এর সারসংক্ষেপ</h3>
    <p>MDP RL-এর মূল framework প্রদান করে। মূল ধারণাগুলো:</p>
    <ul>
      <li><strong>State S</strong> — environment-এর বর্তমান অবস্থা</li>
      <li><strong>Action A</strong> — agent-এর সম্ভাব্য choices</li>
      <li><strong>Reward R</strong> — environment-এর feedback signal</li>
      <li><strong>Markov Property</strong> — future শুধু present-এর উপর নির্ভর করে</li>
      <li><strong>Discount factor γ</strong> — future reward-এর relative importance</li>
      <li><strong>Value function V(s)</strong> — state-এর long-term worth</li>
      <li><strong>Bellman equation</strong> — value function-এর recursive structure</li>
      <li><strong>Optimal policy π*</strong> — সর্বোচ্চ expected return দেওয়া policy</li>
    </ul>
    <p>পরের পর্বে আমরা দেখব কীভাবে Q-Learning algorithm দিয়ে Q-function সরাসরি শেখা যায়, model (transition function) না জেনেও।</p>
  `
};
