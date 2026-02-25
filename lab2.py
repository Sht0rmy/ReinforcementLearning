import numpy as np
import gymnasium as gym
import time

# ══════════════════════════════════════════════════════════════════════════════
# 1. ЗАГАЛЬНІ ПАРАМЕТРИ
# ══════════════════════════════════════════════════════════════════════════════

POSITION_MIN, POSITION_MAX = -1.2, 0.6
VELOCITY_MIN, VELOCITY_MAX = -0.07, 0.07
GOAL_POSITION = 0.5

N_POS = 30
N_VEL = 30
N_STATES  = N_POS * N_VEL
N_ACTIONS = 3

GAMMA = 0.99
THETA = 1e-4

# Monte Carlo
MC_EPISODES     = 5_000   # кількість навчальних епізодів
MC_MAX_STEPS    = 200     # максимальна довжина епізоду
MC_EPSILON_START = 1.0
MC_EPSILON_END   = 0.05
MC_EPSILON_DECAY = 0.9995
MC_ALPHA         = None   # None → averaging; або float для incremental update

# Оцінка
EVAL_EPISODES = 100

position_bins = np.linspace(POSITION_MIN, POSITION_MAX, N_POS)
velocity_bins = np.linspace(VELOCITY_MIN, VELOCITY_MAX, N_VEL)


# ══════════════════════════════════════════════════════════════════════════════
# 2. УТИЛІТИ
# ══════════════════════════════════════════════════════════════════════════════

def discretize(obs):
    pi = np.clip(np.digitize(obs[0], position_bins) - 1, 0, N_POS - 1)
    vi = np.clip(np.digitize(obs[1], velocity_bins) - 1, 0, N_VEL - 1)
    return pi, vi

def s_idx(pi, vi):
    return pi * N_VEL + vi

def mountain_car_step(state, action):
    """Власна детермінована модель динаміки (ідентична Gymnasium)."""
    position, velocity = state
    force, gravity = 0.001, 0.0025
    velocity += (action - 1) * force - gravity * np.cos(3 * position)
    velocity  = np.clip(velocity, VELOCITY_MIN, VELOCITY_MAX)
    position += velocity
    position  = np.clip(position, POSITION_MIN, POSITION_MAX)
    if position == POSITION_MIN and velocity < 0:
        velocity = 0.0
    done   = bool(position >= GOAL_POSITION)
    return np.array([position, velocity]), -1.0, done


# ══════════════════════════════════════════════════════════════════════════════
# 3. ПОБУДОВА P та R  (для DP-методів)
# ══════════════════════════════════════════════════════════════════════════════

def build_model():
    P = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_STATES, N_ACTIONS, N_STATES))

    for pi in range(N_POS):
        for vi in range(N_VEL):
            s = s_idx(pi, vi)
            state = np.array([position_bins[pi], velocity_bins[vi]])

            if state[0] >= GOAL_POSITION:
                for a in range(N_ACTIONS):
                    P[s, a, s] = 1.0
                continue

            for a in range(N_ACTIONS):
                ns_state, reward, done = mountain_car_step(state, a)
                npi, nvi = discretize(ns_state)
                ns = s_idx(npi, nvi)
                P[s, a, ns] = 1.0
                R[s, a, ns] = reward

    return P, R


# ══════════════════════════════════════════════════════════════════════════════
# 4. VALUE ITERATION
# ══════════════════════════════════════════════════════════════════════════════

def value_iteration(P, R):
    V = np.zeros(N_STATES)
    iters = 0

    while True:
        delta = 0.0
        iters += 1
        for s in range(N_STATES):
            v = V[s]
            Q = [np.sum(P[s, a] * (R[s, a] + GAMMA * V)) for a in range(N_ACTIONS)]
            V[s]  = max(Q)
            delta = max(delta, abs(v - V[s]))
        if delta < THETA:
            break

    policy = np.array([np.argmax([np.sum(P[s, a] * (R[s, a] + GAMMA * V))
                                   for a in range(N_ACTIONS)]) for s in range(N_STATES)])
    return V, policy, iters


# ══════════════════════════════════════════════════════════════════════════════
# 5. POLICY ITERATION
# ══════════════════════════════════════════════════════════════════════════════

def policy_iteration(P, R):
    policy = np.zeros(N_STATES, dtype=int)
    V      = np.zeros(N_STATES)
    iters  = 0

    while True:
        iters += 1
        # Evaluation
        while True:
            delta = 0.0
            for s in range(N_STATES):
                v    = V[s]
                a    = policy[s]
                V[s] = np.sum(P[s, a] * (R[s, a] + GAMMA * V))
                delta = max(delta, abs(v - V[s]))
            if delta < THETA:
                break
        # Improvement
        stable = True
        for s in range(N_STATES):
            old = policy[s]
            Q = [np.sum(P[s, a] * (R[s, a] + GAMMA * V)) for a in range(N_ACTIONS)]
            policy[s] = np.argmax(Q)
            if old != policy[s]:
                stable = False
        if stable:
            break

    return V, policy, iters


# ══════════════════════════════════════════════════════════════════════════════
# 6. MONTE CARLO (On-Policy Every-Visit ε-greedy, First-Visit варіант)
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo(episodes=MC_EPISODES):
    """
    On-Policy First-Visit MC Control з ε-greedy стратегією.

    Q[s, a]   — оцінки Q-функції
    N[s, a]   — лічильник відвідувань (для averaging)
    """
    Q       = np.zeros((N_STATES, N_ACTIONS))
    N_count = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)

    epsilon      = MC_EPSILON_START
    episode_lens = []
    success_buf  = []

    env = gym.make("MountainCar-v0")

    for ep in range(episodes):
        # ── генеруємо епізод ──────────────────────────────────────────────
        obs, _ = env.reset()
        episode = []  # [(s, a, r)]

        for _ in range(MC_MAX_STEPS):
            pi_, vi_ = discretize(obs)
            s = s_idx(pi_, vi_)

            # ε-greedy вибір дії
            if np.random.rand() < epsilon:
                a = np.random.randint(N_ACTIONS)
            else:
                a = np.argmax(Q[s])

            obs_next, r, terminated, truncated, _ = env.step(a)
            episode.append((s, a, r))

            if terminated or truncated:
                break
            obs = obs_next

        success_buf.append(1 if r > -1 else 0)   # r=-1 завжди, крім термінації
        # точніша перевірка: якщо terminated (досягнуто мети)
        success_buf[-1] = int(terminated if 'terminated' in dir() else False)
        episode_lens.append(len(episode))

        # ── оновлення Q (First-Visit averaging) ──────────────────────────
        G = 0.0
        visited = set()
        for (s, a, r) in reversed(episode):
            G = r + GAMMA * G
            if (s, a) not in visited:
                visited.add((s, a))
                N_count[s, a] += 1
                # Incremental mean
                Q[s, a] += (G - Q[s, a]) / N_count[s, a]

        # ── decay ε ──────────────────────────────────────────────────────
        epsilon = max(MC_EPSILON_END, epsilon * MC_EPSILON_DECAY)

        if (ep + 1) % 500 == 0:
            succ = np.mean(success_buf[-500:]) * 100
            print(f"  MC  ep {ep+1:5d}/{episodes}  ε={epsilon:.4f}  "
                  f"succ={succ:.1f}%  mean_steps={np.mean(episode_lens[-500:]):.1f}")

    env.close()

    # Жадібна стратегія відносно Q
    policy = np.argmax(Q, axis=1)
    return Q, policy


# ══════════════════════════════════════════════════════════════════════════════
# 7. ОЦІНКА СТРАТЕГІЇ
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(policy, n_episodes=EVAL_EPISODES):
    env = gym.make("MountainCar-v0")
    rewards, successes = [], 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0
        terminated = False
        for _ in range(MC_MAX_STEPS):
            pi_, vi_ = discretize(obs)
            a = policy[s_idx(pi_, vi_)]
            obs, r, terminated, truncated, _ = env.step(a)
            total_r += r
            if terminated or truncated:
                break
        rewards.append(total_r)
        if terminated:
            successes += 1

    env.close()
    return np.mean(rewards), successes / n_episodes * 100


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("Будуємо модель P та R …")
    t0 = time.time()
    P, R = build_model()
    print(f"  Готово за {time.time()-t0:.2f}с")

    print("\n[1/3] Value Iteration …")
    t0 = time.time()
    V_vi, pol_vi, iters_vi = value_iteration(P, R)
    t_vi = time.time() - t0
    r_vi, s_vi = evaluate_policy(pol_vi)
    print(f"  ітерацій={iters_vi}  час={t_vi:.2f}с  reward={r_vi:.1f}  success={s_vi:.0f}%")

    print("\n[2/3] Policy Iteration …")
    t0 = time.time()
    V_pi, pol_pi, iters_pi = policy_iteration(P, R)
    t_pi = time.time() - t0
    r_pi, s_pi = evaluate_policy(pol_pi)
    print(f"  ітерацій={iters_pi}  час={t_pi:.2f}с  reward={r_pi:.1f}  success={s_pi:.0f}%")

    print(f"\n[3/3] Monte Carlo ({MC_EPISODES} епізодів) …")
    t0 = time.time()
    Q_mc, pol_mc = monte_carlo(MC_EPISODES)
    t_mc = time.time() - t0
    r_mc, s_mc = evaluate_policy(pol_mc)
    print(f"  час={t_mc:.2f}с  reward={r_mc:.1f}  success={s_mc:.0f}%")

    print("\n" + "=" * 60)
    print(f"{'Метод':<22} {'Ітерацій':>10} {'Час (с)':>10} {'Reward':>10} {'Успіх %':>10}")
    print("-" * 62)
    print(f"{'Value Iteration':<22} {iters_vi:>10} {t_vi:>10.2f} {r_vi:>10.1f} {s_vi:>9.0f}%")
    print(f"{'Policy Iteration':<22} {iters_pi:>10} {t_pi:>10.2f} {r_pi:>10.1f} {s_pi:>9.0f}%")
    print(f"{'Monte Carlo':<22} {MC_EPISODES:>10} {t_mc:>10.2f} {r_mc:>10.1f} {s_mc:>9.0f}%")

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"{'Метод':<22} {'Ітерацій':>10} {'Час (с)':>10} {'Reward':>10} {'Успіх %':>10}\n")
        f.write("-" * 62 + "\n")
        f.write(f"{'Value Iteration':<22} {iters_vi:>10} {t_vi:>10.2f} {r_vi:>10.1f} {s_vi:>9.0f}%\n")
        f.write(f"{'Policy Iteration':<22} {iters_pi:>10} {t_pi:>10.2f} {r_pi:>10.1f} {s_pi:>9.0f}%\n")
        f.write(f"{'Monte Carlo':<22} {MC_EPISODES:>10} {t_mc:>10.2f} {r_mc:>10.1f} {s_mc:>9.0f}%\n")
