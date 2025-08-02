import wandb
from main.environments.TimeSeriesEnvironment import TimeSeriesEnvironment

wandb.login()


def initialiseWandb(yamlConfig, agent, agentConfig):
    wandb.init(
        project=yamlConfig["project_name"],
        config=agentConfig,
        name=f"exp-{agentConfig['phase']}"
        + ("_NORM" if yamlConfig["normalise_data"] else ""),
        reinit=True,
        group=agentConfig["group"],
        mode=yamlConfig["wandb_state"],
    )

    if wandb.run is not None:
        wandb.watch(agent.actor, log="all")
        wandb.watch(agent.critic, log="all")
        wandb.watch(agent.featureExtractor, log="all")


def getEnv(yamlConfig):
    return TimeSeriesEnvironment(
        TIME_WINDOW=yamlConfig["time_window"],
        startCash=yamlConfig["env"]["start_cash"],
        transactionCost=yamlConfig["env"]["transaction_cost"],
        normaliseData=yamlConfig["normalise_data"],
        perturbationNoise=yamlConfig["env"]["perturbation_noise"],
    )
