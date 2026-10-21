import json

import numpy as np

from .control_env import episode_metrics, run_episode


def relu(x):
    return np.maximum(x, 0.0)


class NumpyDQN:


    def __init__(
        self, state_dim, n_actions, hidden=(64, 64), lr=1e-3, gamma=0.97, seed=0
    ):
        if state_dim < 1 or n_actions < 2 or len(hidden) != 2 or min(hidden) < 1:
            raise ValueError("Invalid network dimensions.")
        if lr <= 0 or not np.isfinite(lr) or not 0 <= gamma <= 1:
            raise ValueError("Invalid learning rate or discount factor.")
        self.state_dim, self.n_actions, self.hidden = (
            state_dim,
            n_actions,
            tuple(hidden),
        )
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.lr = lr
        sizes = [state_dim, hidden[0], hidden[1], n_actions]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])
            self.W.append(self.rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.adam_t = 0

    def forward(self, X):
        z1 = X @ self.W[0] + self.b[0]
        a1 = relu(z1)
        z2 = a1 @ self.W[1] + self.b[1]
        a2 = relu(z2)
        q = a2 @ self.W[2] + self.b[2]
        return q, (X, z1, a1, z2, a2)

    def q(self, X):
        return self.forward(np.atleast_2d(X))[0]

    def act(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q(obs)[0]))

    def get_weights(self):
        return [w.copy() for w in self.W] + [b.copy() for b in self.b]

    def network_config(self):
        return dict(
            architecture="mlp",
            state_dim=self.state_dim,
            n_actions=self.n_actions,
            hidden=list(self.hidden),
            lr=self.lr,
            gamma=self.gamma,
        )

    def new_peer(self, seed=0):
        config = self.network_config()
        config.pop("architecture")
        return type(self)(**config, seed=seed)

    def set_weights(self, weights, reset_optimizer=True):
        shapes = [a.shape for a in self.get_weights()]
        if len(weights) != len(shapes) or any(
            np.shape(w) != shape or not np.isfinite(w).all()
            for w, shape in zip(weights, shapes)
        ):
            raise ValueError("Invalid network weights.")
        n = len(self.W)
        self.W = [w.copy() for w in weights[:n]]
        self.b = [b.copy() for b in weights[n:]]
        if reset_optimizer:
            self.mW = [np.zeros_like(w) for w in self.W]
            self.vW = [np.zeros_like(w) for w in self.W]
            self.mb = [np.zeros_like(b) for b in self.b]
            self.vb = [np.zeros_like(b) for b in self.b]
            self.adam_t = 0

    def train_on_batch(self, states, targets):
        q, (X, z1, a1, z2, a2) = self.forward(states)
        n = states.shape[0]
        dq = (q - targets) / n
        gW2 = a2.T @ dq
        gb2 = dq.sum(0)
        da2 = dq @ self.W[2].T
        da2[z2 <= 0] = 0
        gW1 = a1.T @ da2
        gb1 = da2.sum(0)
        da1 = da2 @ self.W[1].T
        da1[z1 <= 0] = 0
        gW0 = X.T @ da1
        gb0 = da1.sum(0)
        self._adam([gW0, gW1, gW2], [gb0, gb1, gb2])
        return float(np.sum((q - targets) ** 2) / (2 * n))

    def _adam(self, gW, gb, beta1=0.9, beta2=0.999, eps=1e-8):
        self.adam_t += 1
        t = self.adam_t
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * gW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * gW[i] ** 2
            mhat = self.mW[i] / (1 - beta1**t)
            vhat = self.vW[i] / (1 - beta2**t)
            self.W[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * gb[i] ** 2
            mhb = self.mb[i] / (1 - beta1**t)
            vhb = self.vb[i] / (1 - beta2**t)
            self.b[i] -= self.lr * mhb / (np.sqrt(vhb) + eps)


class DuelingDQN(NumpyDQN):


    def __init__(
        self,
        state_dim=19,
        n_actions=8,
        hidden=(128, 128),
        lr=1e-4,
        gamma=0.95,
        seed=0,
        stream_hidden=64,
        gradient_clip=5.0,
    ):
        super().__init__(state_dim, n_actions, hidden, lr, gamma, seed)
        if not isinstance(stream_hidden, (int, np.integer)) or stream_hidden < 1:
            raise ValueError("stream_hidden must be a positive integer.")
        if gradient_clip is not None and (
            not np.isfinite(gradient_clip) or gradient_clip <= 0
        ):
            raise ValueError("gradient_clip must be positive or None.")
        self.stream_hidden = stream_hidden
        self.gradient_clip = gradient_clip
        shapes = [
            (state_dim, hidden[0]),
            (hidden[0], hidden[1]),
            (hidden[1], stream_hidden),
            (stream_hidden, 1),
            (hidden[1], stream_hidden),
            (stream_hidden, n_actions),
        ]
        self.W = [self.rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in shapes]
        self.b = [np.zeros(b) for _, b in shapes]
        self.set_weights(self.get_weights())

    def network_config(self):
        return dict(
            super().network_config(),
            architecture="dueling",
            stream_hidden=self.stream_hidden,
            gradient_clip=self.gradient_clip,
        )

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.state_dim or not np.isfinite(X).all():
            raise ValueError("Expected finite states with shape (batch, state_dim).")
        z1 = X @ self.W[0] + self.b[0]
        a1 = relu(z1)
        z2 = a1 @ self.W[1] + self.b[1]
        a2 = relu(z2)
        zv = a2 @ self.W[2] + self.b[2]
        av = relu(zv)
        value = av @ self.W[3] + self.b[3]
        za = a2 @ self.W[4] + self.b[4]
        aa = relu(za)
        advantage = aa @ self.W[5] + self.b[5]
        q = value + advantage - advantage.mean(axis=1, keepdims=True)
        return q, (X, z1, a1, z2, a2, zv, av, za, aa)

    def loss_and_gradients(self, states, targets):
        q, (X, z1, a1, z2, a2, zv, av, za, aa) = self.forward(states)
        targets = np.asarray(targets, dtype=float)
        if targets.shape != q.shape or not np.isfinite(targets).all() or len(X) == 0:
            raise ValueError("Targets must match the nonempty finite Q batch.")
        dq = 2 * (q - targets) / len(X)
        dv = dq.sum(axis=1, keepdims=True)
        da = dq - dq.mean(axis=1, keepdims=True)
        dzv = (dv @ self.W[3].T) * (zv > 0)
        dza = (da @ self.W[5].T) * (za > 0)
        dz2 = (dzv @ self.W[2].T + dza @ self.W[4].T) * (z2 > 0)
        dz1 = (dz2 @ self.W[1].T) * (z1 > 0)
        gW = [X.T @ dz1, a1.T @ dz2, a2.T @ dzv, av.T @ dv, a2.T @ dza, aa.T @ da]
        gb = [g.sum(axis=0) for g in (dz1, dz2, dzv, dv, dza, da)]
        return float(np.sum((q - targets) ** 2) / len(X)), gW, gb

    def train_on_batch(self, states, targets):
        loss, gW, gb = self.loss_and_gradients(states, targets)
        norm = np.sqrt(sum(np.sum(g * g) for g in gW + gb))
        if not np.isfinite(norm):
            raise ValueError("Nonfinite Dueling DQN gradient.")
        if self.gradient_clip is not None and norm > self.gradient_clip:
            factor = self.gradient_clip / norm
            gW, gb = [g * factor for g in gW], [g * factor for g in gb]
        self._adam(gW, gb)
        return loss


class Replay:
    def __init__(self, state_dim, cap=20000, seed=0):
        if cap < 1 or state_dim < 1:
            raise ValueError("Invalid replay dimensions.")
        self.cap = cap
        self.s = np.zeros((cap, state_dim))
        self.a = np.zeros(cap, int)
        self.r = np.zeros(cap)
        self.ns = np.zeros((cap, state_dim))
        self.d = np.zeros(cap)
        self.i = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, ns, d):
        k = self.i
        self.s[k] = s
        self.a[k] = a
        self.r[k] = r
        self.ns[k] = ns
        self.d[k] = d
        self.i = (self.i + 1) % self.cap
        if self.i == 0:
            self.full = True

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, bs):
        idx = self.rng.integers(0, len(self), bs)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx]


def dqn_policy(agent):

    return lambda obs, raw: int(np.argmax(agent.q(obs)[0]))


class TrainingState:


    def __init__(self, agent, seed, eps_start, replay_capacity, schedule):
        self.replay = Replay(agent.state_dim, cap=replay_capacity, seed=seed)
        self.target = agent.new_peer(seed=seed + 1)
        self.target.set_weights(agent.get_weights())
        self.steps = 0
        self.updates = 0
        self.epsilon = eps_start
        self.schedule = schedule


def _check_validation(train_envs, val_envs):
    for train in train_envs.values():
        for val in val_envs.values():
            if train is val:
                raise ValueError(
                    "Training and validation environments must be separate."
                )
            if (
                train.label == val.label
                and train.frame.time.max() >= val.frame.time.min()
            ):
                raise ValueError(
                    "Validation must follow training without overlapping timestamps."
                )


def evaluate_agent(agent, envs):

    details = {}
    for label, env in envs.items():
        starts = sorted(
            set(env.starts[int((len(env.starts) - 1) * f)] for f in (0.0, 0.5, 1.0))
        )
        rolls = [run_episode(env, dqn_policy(agent), s) for s in starts]
        details[label] = dict(
            mean_reward=float(np.mean([np.sum(r["rewards"]) for r in rolls])),
            windows=[episode_metrics(r) for r in rolls],
            starts=starts,
        )
    return dict(
        mean_reward=float(np.mean([x["mean_reward"] for x in details.values()])),
        stations=details,
    )


def train_dqn(
    env,
    episodes=180,
    batch=64,
    target_sync=300,
    reward_scale=0.1,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.985,
    warmup=500,
    seed=0,
    agent=None,
    log=True,
    lr=None,
    gamma=None,
    validation_env=None,
    replay_capacity=20000,
    eps_decay_unit="episode",
    target_sync_unit="environment_step",
):
    if (
        min(episodes, batch, target_sync) < 1
        or warmup < 0
        or not 0 <= eps_end <= eps_start <= 1
        or not 0 < eps_decay <= 1
    ):
        raise ValueError("Invalid DQN schedule.")
    if eps_decay_unit not in {
        "episode",
        "environment_step",
    } or target_sync_unit not in {
        "environment_step",
        "optimizer_step",
    }:
        raise ValueError("Unknown exploration or target update time unit.")
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("reward_scale must be positive.")
    if agent is None:
        agent = NumpyDQN(
            env.obs_dim,
            env.n_actions,
            lr=1e-3 if lr is None else lr,
            gamma=0.97 if gamma is None else gamma,
            seed=seed,
        )
    if (agent.state_dim, agent.n_actions) != (env.obs_dim, env.n_actions):
        raise ValueError("Agent and environment dimensions differ.")
    if lr is not None:
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive.")
        agent.lr = lr
    if gamma is not None:
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0,1].")
        agent.gamma = gamma
    if validation_env is not None:
        _check_validation({env.label: env}, {validation_env.label: validation_env})
    schedule = (
        replay_capacity,
        eps_decay_unit,
        eps_end,
        eps_decay,
        target_sync_unit,
        target_sync,
    )
    if not hasattr(agent, "_training"):
        agent._training = TrainingState(
            agent, seed, eps_start, replay_capacity, schedule
        )
    state = agent._training
    if state.schedule != schedule:
        raise ValueError("Cannot change a persistent training schedule implicitly.")
    history = []
    for ep in range(episodes):
        obs, done, reward_sum, losses, steps = env.reset(), False, 0.0, [], 0
        while not done:
            action = agent.act(obs, state.epsilon)
            next_obs, reward, done, _ = env.step(action)
            state.replay.push(obs, action, reward * reward_scale, next_obs, float(done))
            obs, reward_sum, steps = next_obs, reward_sum + reward, steps + 1
            state.steps += 1
            if len(state.replay) >= max(batch, warmup):
                x, a, r, nx, terminal = state.replay.sample(batch)
                targets = agent.q(x).copy()
                targets[np.arange(batch), a] = r + agent.gamma * state.target.q(nx).max(
                    1
                ) * (1 - terminal)
                losses.append(agent.train_on_batch(x, targets))
                state.updates += 1
                if (
                    target_sync_unit == "optimizer_step"
                    and state.updates % target_sync == 0
                ):
                    state.target.set_weights(agent.get_weights())
            if (
                target_sync_unit == "environment_step"
                and state.steps % target_sync == 0
            ):
                state.target.set_weights(agent.get_weights())
            if eps_decay_unit == "environment_step":
                state.epsilon = max(eps_end, state.epsilon * eps_decay)
        if eps_decay_unit == "episode":
            state.epsilon = max(eps_end, state.epsilon * eps_decay)
        row = dict(
            episode=ep + 1,
            train_reward=reward_sum,
            train_steps=steps,
            loss=float(np.mean(losses)) if losses else None,
            epsilon=state.epsilon,
            cumulative_steps=state.steps,
            cumulative_updates=state.updates,
            replay_size=len(state.replay),
        )
        if validation_env is not None:
            row["validation"] = evaluate_agent(
                agent, {validation_env.label: validation_env}
            )
        history.append(row)
        if log and (ep % 20 == 0 or ep == episodes - 1):
            print(f"episode={ep + 1} train_reward={reward_sum:.3f} steps={state.steps}")
    return agent, history


def _check_envs(envs):
    if not envs:
        raise ValueError("No environments.")
    first = next(iter(envs.values()))
    for env in envs.values():
        if (env.obs_dim, env.n_actions, env.max_steps) != (
            first.obs_dim,
            first.n_actions,
            first.max_steps,
        ):
            raise ValueError(
                "Network dimensions and episode budgets must agree across clients."
            )
        if (
            env.spec.metadata() != first.spec.metadata()
            or env.cost_weight != first.cost_weight
            or not np.array_equal(env.norm, first.norm)
        ):
            raise ValueError(
                "Federated clients require the same action semantics, objective and feature normalization."
            )
    return first


def fedavg_train(
    envs,
    rounds=18,
    local_episodes=10,
    seed=0,
    agg="fedavg",
    log=True,
    validation_envs=None,
    agent=None,
    **kw,
):
    if agg != "fedavg":
        raise ValueError(
            "Only FedAvg is supported. Reward-based quality weights and unverified DP were removed."
        )
    if rounds < 1 or local_episodes < 1:
        raise ValueError("Positive training budgets are required.")
    first = _check_envs(envs)
    if validation_envs is not None:
        _check_validation(envs, validation_envs)
    global_agent = (
        agent
        if agent is not None
        else NumpyDQN(
            first.obs_dim,
            first.n_actions,
            seed=seed,
            lr=kw.get("lr", 1e-3),
            gamma=kw.get("gamma", 0.97),
        )
    )
    if (global_agent.state_dim, global_agent.n_actions) != (
        first.obs_dim,
        first.n_actions,
    ):
        raise ValueError("Agent and client dimensions differ.")
    if "lr" in kw and kw["lr"] is not None:
        if not np.isfinite(kw["lr"]) or kw["lr"] <= 0:
            raise ValueError("lr must be finite and positive.")
        global_agent.lr = kw["lr"]
    if "gamma" in kw and kw["gamma"] is not None:
        if not 0 <= kw["gamma"] <= 1:
            raise ValueError("gamma must be in [0,1].")
        global_agent.gamma = kw["gamma"]
    local_agents, history = {}, []
    for rd in range(rounds):
        params, steps = [], []
        for i, (label, env) in enumerate(envs.items()):
            if label not in local_agents:
                local_agents[label] = global_agent.new_peer(seed=seed + i + 1)
            local = local_agents[label]
            local.set_weights(global_agent.get_weights())
            if hasattr(local, "_training"):
                local._training.target.set_weights(global_agent.get_weights())
            local, logs = train_dqn(
                env,
                episodes=local_episodes,
                agent=local,
                log=False,
                seed=seed + i,
                **kw,
            )
            params.append(local.get_weights())
            steps.append(sum(x["train_steps"] for x in logs))
        weights = np.asarray(steps) / sum(steps)
        global_agent.set_weights(
            [
                sum(w * p[j] for w, p in zip(weights, params))
                for j in range(len(params[0]))
            ]
        )
        row = dict(
            round=rd + 1, client_train_steps=steps, aggregation_weights=weights.tolist()
        )
        if validation_envs is not None:
            row["global_validation"] = evaluate_agent(global_agent, validation_envs)
        history.append(row)
        if log:
            print(f"round={rd + 1} client_train_steps={steps}")
    return global_agent, history


def train_centralized_dqn(
    envs, cycles=20, local_episodes=10, seed=0, log=True, agent=None, **kw
):
    _check_envs(envs)
    if cycles < 1 or local_episodes < 1:
        raise ValueError("Positive training budgets are required.")
    for cycle in range(cycles):
        for env in envs.values():
            agent, _ = train_dqn(
                env, episodes=local_episodes, agent=agent, log=False, seed=seed, **kw
            )
        if log:
            print(f"cycle={cycle + 1} shared_replay_size={len(agent._training.replay)}")
    return agent


def save_agent(path, agent, metadata):

    required = {
        "normalization",
        "control_spec",
        "input_fingerprints",
        "training",
        "transition_sha256",
    }
    if not required <= set(metadata):
        raise ValueError(
            "Checkpoint requires normalization, control specification, fingerprints and training metadata."
        )
    meta = dict(
        metadata,
        schema_version=3,
        network=agent.network_config(),
        checkpoint_type="inference",
    )
    np.savez(
        path,
        **{f"w{i}": w for i, w in enumerate(agent.get_weights())},
        metadata=json.dumps(meta, ensure_ascii=False),
    )


def load_agent(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta["schema_version"] == 2:
            config = {
                key: meta[key]
                for key in ("state_dim", "n_actions", "hidden", "lr", "gamma")
            }
            agent = NumpyDQN(**config)
        elif meta["schema_version"] == 3:
            config = dict(meta["network"])
            architecture = config.pop("architecture")
            if architecture not in {"mlp", "dueling"}:
                raise ValueError("Unknown Q-network architecture.")
            agent = {"mlp": NumpyDQN, "dueling": DuelingDQN}[architecture](**config)
        else:
            raise ValueError("Unsupported checkpoint schema.")
        agent.set_weights([data[f"w{i}"] for i in range(len(agent.get_weights()))])
    return agent, meta
