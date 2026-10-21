from .control_train import get_station_agents
from .dqn import fedavg_train, train_centralized_dqn


def get_paradigm_agents(
    train_envs, validation_envs=None, rounds=20, local_episodes=10, seed=0, **kw
):


    import copy

    states = {
        lb: copy.deepcopy(env.rng.bit_generator.state) for lb, env in train_envs.items()
    }

    def restore():
        for lb, env in train_envs.items():
            env.rng.bit_generator.state = copy.deepcopy(states[lb])

    local, logs = get_station_agents(
        train_envs, validation_envs, episodes=rounds * local_episodes, seed=seed, **kw
    )
    restore()
    centralized = train_centralized_dqn(
        train_envs, cycles=rounds, local_episodes=local_episodes, seed=seed, **kw
    )
    restore()
    federated, fed_logs = fedavg_train(
        train_envs,
        rounds=rounds,
        local_episodes=local_episodes,
        validation_envs=validation_envs,
        seed=seed,
        **kw,
    )
    return dict(local=local, centralized=centralized, federated=federated), dict(
        local=logs, federated=fed_logs
    )


def get_loso_agent(source_train_envs, source_validation_envs=None, **training):

    return fedavg_train(
        source_train_envs, validation_envs=source_validation_envs, **training
    )
