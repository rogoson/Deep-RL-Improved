import wandb
from environments.TimeSeriesEnvironment import TimeSeriesEnvironment

wandb.login()


def initialiseWandb(yamlConfig, agent, agentConfig):
    cfg = agentConfig
    wandb.init(
        project=yamlConfig["project_name"],
        config=cfg,
        name=f"exp-{agentConfig['phase']}"
        + ("_NORM" if yamlConfig["normalise_data"] else ""),
        reinit=True,
        group=cfg["group"],
        mode="disabled",
    )

    if wandb.run is not None:
        wandb.watch(agent.actor, log="all")
        wandb.watch(agent.critic, log="all")
        wandb.watch(agent.featureExtractor, log="all")


def getEnv(yamlConfig):
    return TimeSeriesEnvironment(
        TIME_WINDOW=yamlConfig["time_window"],
        startCash=yamlConfig["start_cash"],
        transactionCost=yamlConfig["transaction_cost"],
        normaliseData=yamlConfig["normalise_data"],
        perturbationNoise=yamlConfig["perturbation_noise"],
    )
