from main.utils.GeneralUtils import seed
from main.trainingAndEval.Training import trainingLoop
from .ExperimentsFunctions import runExperimentFunction, ensureMarketConfig
from .InitialisationHelpers import getEnv
from main.utils.EvaluationConfig import setUpEvaluationConfig
from main.trainingAndEval.Evaluation import evaluateAgent
import os
import yaml
from pathlib import Path
from .TestExperimentsPlotting import (
    plotLearningCurves,
    tabulateBestTestSetPerformance,
    bestPerformancesAndStandardDeviations,
    meanStatisticsTabulated,
    finalIndexComparisonPlot,
)
import wandb
import numpy as np


def mergeTunedHyperparameters(volumeConfigDir):
    """
    The hyperparameter container writes its tuned values into the config on the shared volume.
    Only those tuned values are taken from it - everything else (tickers, dates, ...) comes from
    the freshly built config, so a stale config left on the volume can never override it.
    """
    market = os.environ["MARKET_INDEX"]
    volumeConfigPath = volumeConfigDir / f"temp_config_{market}.yaml"
    configPath = ensureMarketConfig(market)
    if not volumeConfigPath.exists():
        print("No tuned hyperparameters on the volume - using config defaults.")
        return
    with open(volumeConfigPath) as f:
        volumeConfig = yaml.safe_load(f)
    with open(configPath) as f:
        config = yaml.safe_load(f)
    for agentType, agentSettings in volumeConfig.get("agent", {}).items():
        if isinstance(agentSettings, dict) and "best_feature_output_sizes" in agentSettings:
            tuned = agentSettings["best_feature_output_sizes"]
            config["agent"].setdefault(agentType, {})["best_feature_output_sizes"] = tuned
            print(f"Using tuned feature output sizes for {agentType}: {tuned}")
    with open(configPath, "w") as f:
        yaml.dump(config, f)


def getRandomMetrics(yamlConfig, randomRepeats=1000):
    """
    Returns metrics for a random agent.
    """
    variedBaseSeeds = yamlConfig["varied_base_seeds"]
    # split the repeats evenly across the base seeds
    repeatsPerSeed = max(1, randomRepeats // len(variedBaseSeeds))
    experimentConfig = setUpEvaluationConfig(yamlConfig, "random")
    averageRandomPerformance = []
    env = getEnv(yamlConfig)
    env.setup(yamlConfig)
    for rep in range(randomRepeats):
        seedIndex = rep // repeatsPerSeed
        if rep % repeatsPerSeed == 0 and seedIndex < len(variedBaseSeeds):
            BASE_SEED = variedBaseSeeds[seedIndex]
            seed(BASE_SEED)  # Seed the random agent with the base seed
            env.baseSeed = BASE_SEED
        randomArray = evaluateAgent(
            agent=None, env=env, num=0, conf=None, epoch=rep, **experimentConfig
        )
        averageRandomPerformance.append(randomArray)
    averageRandomReturn = (
        np.mean(np.array(averageRandomPerformance), axis=0)[-1]
        / yamlConfig["env"]["start_cash"]
        - 1
    )
    forStd = np.std(
        np.array(averageRandomPerformance)[:, -1] / yamlConfig["env"]["start_cash"] - 1
    )  # CHECK THIS!
    averageRandomPerformance = np.mean(np.array(averageRandomPerformance), axis=0)

    return {
        "average_random_performance": averageRandomPerformance,
        "average_random_return": averageRandomReturn,
        "random_std": forStd,
    }


def testMetricsAndGraphs(yamlConfig, rewards, envDetails, agentType="ppo"):
    randomMetrics = getRandomMetrics(
        yamlConfig, randomRepeats=yamlConfig.get("random_repeats", 1000)
    )
    bestTestSetPerformance = plotLearningCurves(
        yamlConfig=yamlConfig,
        randomMetrics=randomMetrics,
        rewardFunctions=rewards,
        agentType=agentType,
        sumTestTrainingPeriods=envDetails["sum_test_training_periods"],
    )
    tabulateBestTestSetPerformance(
        yamlConfig, bestTestSetPerformance, rewards, randomMetrics, agentType=agentType
    )
    bestPerformancesAndStandardDeviations(
        yamlConfig,
        bestTestSetPerformance,
        randomMetrics["average_random_performance"],
        rewards,
        agentType=agentType,
    )
    meanStatisticsTabulated(
        yamlConfig=yamlConfig,
        bestTestSetPerformance=bestTestSetPerformance,
        avRandReturn=randomMetrics["average_random_return"],
        rewards=rewards,
        agentType=agentType,
    )

    experimentConfig = setUpEvaluationConfig(yamlConfig, "nonRLComparisonStrategies")
    env = getEnv(yamlConfig)
    env.setup(yamlConfig)
    comparisonStrategies = yamlConfig["env"]["comparisonStrategies"]
    benchmarkportfolioValues = dict()
    for strat in comparisonStrategies:
        experimentConfig["comparisonStrat"] = strat.upper() + " Buy-and-Hold"
        benchmarkportfolioValues[strat] = evaluateAgent(
            agent=None,
            env=env,
            num=0,
            conf=None,
            **experimentConfig,
        )

    finalIndexComparisonPlot(
        yamlConfig,
        bestTestSetPerformance,
        randomMetrics["average_random_performance"],
        benchmarkPortVals=benchmarkportfolioValues,
        rewards=rewards,
        agentType=agentType,
    )


# this will actually test them too lol
def trainTestingAgents(yamlConfig, agentType="ppo", phase="reward_testing"):
    REWARDS = {
        "Reward": [
            "Standard Logarithmic Returns",
            "Differential Sharpe Ratio_0.01",
            "Differential Sharpe Ratio_0.05",
            "Differential Sharpe Ratio_0.1",
            "CVaR_0.25",  # booty
            # "CVaR_0.5",
            # "CVaR_1.0",
            # "CVaR_1.5",
            # "CVaR_2.0",
        ]
    }
    if yamlConfig.get("reward_subset"):  # quick test only
        REWARDS["Reward"] = yamlConfig["reward_subset"]
    for extractor in [True, False]:
        yamlConfig["usingLSTMFeatureExtractor"] = extractor
        for s in yamlConfig["varied_base_seeds"]:
            BASE_SEED = s
            seed(BASE_SEED)  # Seed agent initialization
            yamlConfig["env"]["base_seed"] = s
            for rew in REWARDS["Reward"]:
                env = getEnv(yamlConfig)
                trainingLoop(
                    yamlConfig=yamlConfig,
                    env=env,
                    agentType=agentType,
                    conf="Reward Function-" + rew + " | " + f"Strategy-{agentType}",
                    stage=phase,
                    optionalHyperConfig={"reward_function": rew},
                )
                wandb.finish()

        testMetricsAndGraphs(
            yamlConfig,
            rewards=REWARDS["Reward"],
            envDetails=env.datasetsAndDetails,
            agentType=agentType,
        )


if __name__ == "__main__":
    dataDir = Path(__file__).resolve().parent.parent.parent.parent / "data"

    mergeTunedHyperparameters(dataDir / "configs")

    runExperimentFunction(trainTestingAgents)
