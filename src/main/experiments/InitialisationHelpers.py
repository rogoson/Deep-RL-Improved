import wandb
from main.environments.TimeSeriesEnvironment import TimeSeriesEnvironment
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
wandb.login(key=API_KEY)


def initialiseWandb(yamlConfig, agent, agentConfig):
    wandb.init(
        project=yamlConfig["project_name"],
        config=agentConfig,
        name=f"exp-{agentConfig['phase']}"
        + ("_NORM" if yamlConfig["normalise_data"] else ""),
        reinit=True,
        group=f"{agentConfig['strategy']} , {'LSTM' if yamlConfig['usingLSTMFeatureExtractor'] else 'CNN'}Feature | {agentConfig['group']} | {yamlConfig['active_index'].upper()}",
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
