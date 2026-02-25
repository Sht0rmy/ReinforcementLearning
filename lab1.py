import numpy as np
import gymnasium as gym
import time

# ==========================================================
# 1. ПАРАМЕТРИ
# ==========================================================

position_min, position_max = -1.2, 0.6
velocity_min, velocity_max = -0.07, 0.07
goal_position = 0.5

n_pos = 30
n_vel = 30

gamma = 0.99
theta = 1e-4

position_bins = np.linspace(position_min, position_max, n_pos)
velocity_bins = np.linspace(velocity_min, velocity_max, n_vel)

n_states = n_pos * n_vel
n_actions = 3


# ==========================================================
# 2. ДИСКРЕТИЗАЦІЯ
# ==========================================================

def discretize_state(state):
    pos, vel = state

    pos_idx = np.digitize(pos, position_bins) - 1
    vel_idx = np.digitize(vel, velocity_bins) - 1

    pos_idx = np.clip(pos_idx, 0, n_pos - 1)
    vel_idx = np.clip(vel_idx, 0, n_vel - 1)

    return pos_idx, vel_idx


def state_to_index(pos_idx, vel_idx):
    return pos_idx * n_vel + vel_idx


# ==========================================================
# 3. ДИНАМІКА MountainCar
# ==========================================================

def mountain_car_step(state, action):
    position, velocity = state

    force = 0.001
    gravity = 0.0025

    action_force = action - 1  # 0=-1, 1=0, 2=+1

    velocity += action_force * force - gravity * np.cos(3 * position)
    velocity = np.clip(velocity, velocity_min, velocity_max)

    position += velocity
    position = np.clip(position, position_min, position_max)

    if position == position_min and velocity < 0:
        velocity = 0

    done = position >= goal_position
    reward = -1.0

    return np.array([position, velocity]), reward, done


# ==========================================================
# 4. ПОБУДОВА P ТА R (з terminal state)
# ==========================================================

P = np.zeros((n_states, n_actions, n_states))
R = np.zeros((n_states, n_actions, n_states))

for pos_idx in range(n_pos):
    for vel_idx in range(n_vel):

        s_idx = state_to_index(pos_idx, vel_idx)

        state = np.array([
            position_bins[pos_idx],
            velocity_bins[vel_idx]
        ])

        # Якщо це цільовий стан → робимо його поглинаючим
        if state[0] >= goal_position:
            for a in range(n_actions):
                P[s_idx, a, s_idx] = 1.0
                R[s_idx, a, s_idx] = 0.0
            continue

        for a in range(n_actions):

            next_state, reward, done = mountain_car_step(state, a)

            if done:
                # перехід у поглинаючий стан
                npos, nvel = discretize_state(next_state)
                ns_idx = state_to_index(npos, nvel)

                P[s_idx, a, ns_idx] = 1.0
                R[s_idx, a, ns_idx] = reward
            else:
                npos, nvel = discretize_state(next_state)
                ns_idx = state_to_index(npos, nvel)

                P[s_idx, a, ns_idx] = 1.0
                R[s_idx, a, ns_idx] = reward


# ==========================================================
# 5. VALUE ITERATION
# ==========================================================

def value_iteration():
    V = np.zeros(n_states)
    iterations = 0

    while True:
        delta = 0
        iterations += 1

        for s in range(n_states):
            v = V[s]

            Q_values = np.zeros(n_actions)
            for a in range(n_actions):
                Q_values[a] = np.sum(
                    P[s, a] * (R[s, a] + gamma * V)
                )

            V[s] = np.max(Q_values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        Q_values = np.zeros(n_actions)
        for a in range(n_actions):
            Q_values[a] = np.sum(
                P[s, a] * (R[s, a] + gamma * V)
            )
        policy[s] = np.argmax(Q_values)

    return V, policy, iterations


# ==========================================================
# 6. POLICY ITERATION
# ==========================================================

def policy_iteration():
    policy = np.zeros(n_states, dtype=int)
    V = np.zeros(n_states)
    iterations = 0

    while True:
        iterations += 1

        # Policy Evaluation
        while True:
            delta = 0
            for s in range(n_states):
                v = V[s]
                a = policy[s]
                V[s] = np.sum(P[s, a] * (R[s, a] + gamma * V))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # Policy Improvement
        stable = True
        for s in range(n_states):
            old_action = policy[s]

            Q_values = np.zeros(n_actions)
            for a in range(n_actions):
                Q_values[a] = np.sum(
                    P[s, a] * (R[s, a] + gamma * V)
                )

            policy[s] = np.argmax(Q_values)

            if old_action != policy[s]:
                stable = False

        if stable:
            break

    return V, policy, iterations


# ==========================================================
# 7. ОЦІНКА ПОЛІТИКИ
# ==========================================================

def evaluate_policy(policy, episodes=20):
    env = gym.make("MountainCar-v0")

    rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(200):
            pos_idx, vel_idx = discretize_state(state)
            s_idx = state_to_index(pos_idx, vel_idx)

            action = policy[s_idx]
            state, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return np.mean(rewards)


# ==========================================================
# 8. ПОРІВНЯННЯ
# ==========================================================

print("Running Value Iteration...")
start = time.time()
V_vi, policy_vi, iter_vi = value_iteration()
time_vi = time.time() - start
reward_vi = evaluate_policy(policy_vi)

print("Running Policy Iteration...")
start = time.time()
V_pi, policy_pi, iter_pi = policy_iteration()
time_pi = time.time() - start
reward_pi = evaluate_policy(policy_pi)

print("\n========== RESULTS ==========")
print(f"Value Iteration:")
print(f"  Iterations: {iter_vi}")
print(f"  Time: {time_vi:.3f} sec")
print(f"  Avg Reward: {reward_vi:.2f}")

print(f"\nPolicy Iteration:")
print(f"  Iterations: {iter_pi}")
print(f"  Time: {time_pi:.3f} sec")
print(f"  Avg Reward: {reward_pi:.2f}")