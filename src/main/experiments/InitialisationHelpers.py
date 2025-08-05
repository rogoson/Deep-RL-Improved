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
        group=f"{agentConfig['strategy']} | {agentConfig['group']}",
        mode=yamlConfig["wandb_state"],
    )

    if wandb.run is not None:
        if agentConfig["strategy"] == "PPOLSTM":
            wandb.watch(agent.actor, log="all")
            wandb.watch(agent.critic, log="all")
            wandb.watch(agent.featureExtractor, log="all")
        elif agentConfig["strategy"] == "TD3":
            wandb.watch(agent.actor, log="all")
            wandb.watch(agent.critic, log="all")
            wandb.watch(agent.critic2, log="all")
            wandb.watch(agent.featureExtractor, log="all")
            wandb.watch(agent.targetActor, log="all")
            wandb.watch(agent.targetCritic, log="all")
            wandb.watch(agent.targetCritic2, log="all")
            wandb.watch(agent.targetFeatureExtractor, log="all")


def getEnv(yamlConfig):
    return TimeSeriesEnvironment(
        TIME_WINDOW=yamlConfig["time_window"],
        startCash=yamlConfig["env"]["start_cash"],
        transactionCost=yamlConfig["env"]["transaction_cost"],
        normaliseData=yamlConfig["normalise_data"],
        perturbationNoise=yamlConfig["env"]["perturbation_noise"],
    )
