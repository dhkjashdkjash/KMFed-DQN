from .common import load_all_stations
from .control_env import WaterControlEnv, station_series
from .dqn import fedavg_train, train_dqn


def make_environments(
    data_dir, train_end, validation_end, spec, max_steps=200, cost_weight=1.0, seed=0
):
    stations, _ = load_all_stations(data_dir)
    split = station_series(stations, train_end, validation_end)
    result = {part: {} for part in ("train", "validation", "test")}
    for i, (label, parts) in enumerate(split.items()):
        for part in result:
            result[part][label] = WaterControlEnv(
                parts[part],
                parts["norm"],
                spec,
                max_steps=max_steps,
                cost_weight=cost_weight,
                seed=seed + i,
                label=label,
            )
    return result


def get_station_agents(train_envs, validation_envs=None, **training):
    agents, histories = {}, {}
    for label, env in train_envs.items():
        validation = None if validation_envs is None else validation_envs[label]
        agents[label], histories[label] = train_dqn(
            env, validation_env=validation, **training
        )
    return agents, histories


def get_federated_agent(train_envs, validation_envs=None, **training):
    return fedavg_train(train_envs, validation_envs=validation_envs, **training)
