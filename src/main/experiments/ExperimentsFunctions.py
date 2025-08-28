from main.utils.GeneralUtils import seed, getFileWritingLocation
from main.trainingAndEval.Training import trainingLoop
from .NonTestExperimentsPlotting import (
    runParameterComparison,
)
from .InitialisationHelpers import getEnv
from main.utils.EvaluationConfig import setUpEvaluationConfig
from main.trainingAndEval.Evaluation import evaluateAgent
from pathlib import Path
from .TestExperimentsPlotting import (
    plotLearningCurves,
    tabulateBestTestSetPerformance,
    bestPerformancesAndStandardDeviations,
    meanStatisticsTabulated,
    finalIndexComparisonPlot,
)
import os
import wandb
import numpy as np


def hyperparameterTuning(yamlConfig, agentType="ppo", phase="hyperparameter_tuning"):
    """
    Runs the hyperparameter sweep by sequentially activating one test type at a time.
    """
    TESTING = {
        "FEATURE OUTPUT SIZE": False,
        "NORMALIZE DATA": False,
    }
    portfolioValues = dict()
    for extractor in [True, False]:
        yamlConfig["usingLSTMFeatureExtractor"] = extractor
        for key in list(TESTING.keys()):
            print("=" * 50)
            TESTING[key] = True

            sweepParams = {
                "FEATURE OUTPUT SIZE": {
                    "values": yamlConfig["hyperparameters"]["feature_output_sizes"],
                    "overrides": {"feature_output_size": None},
                },
                "NORMALIZE DATA": {  # not really a hyperparam but easier than a whole other experiment
                    "values": [True, False],
                    "overrides": {},
                },
            }

            # Iterate over active test types defined in TESTING
            for s in yamlConfig["varied_base_seeds"]:
                BASE_SEED = s
                seed(BASE_SEED)
                yamlConfig["env"]["base_seed"] = s
                for testType, active in TESTING.items():
                    if active and testType in sweepParams:
                        param_info = sweepParams[testType]
                        for value in param_info["values"]:
                            print(f"Running sweep for {testType}: {value}")
                            # Build the overrides dict, substituting sweep values where needed.
                            overrides = {
                                key: (value if override is None else override)
                                for key, override in param_info["overrides"].items()
                            }
                            if testType == "NORMALIZE DATA":
                                yamlConfig["normalise_data"] = value
                            env = getEnv(yamlConfig)
                            portfolioValues[s] = trainingLoop(
                                yamlConfig,
                                env,
                                agentType,
                                stage=phase,
                                conf=f"{testType.lower().title()} - {value} | Strategy-{agentType}",
                                optionalHyperConfig=overrides,
                            )
                            baseFolder = getFileWritingLocation(
                                yamlConfig, agentType=agentType
                            )
                            desiredFolder = (
                                f"{baseFolder}/portfolios/{testType}/{value}/{s}/"
                            )
                            if not os.path.exists(desiredFolder):
                                os.makedirs(desiredFolder)
                            np.savetxt(
                                f"{desiredFolder}validationPerformances.txt",
                                portfolioValues[s]["validation_performances"],
                                fmt="%f",
                            )
                            wandb.finish()
                        break  # Run only one active test type per sweep
            TESTING[key] = False
            print("=" * 50)
    runParameterComparison(yamlConfig, env, agentType)


def getRandomMetrics(yamlConfig, randomRepeats=1000):
    """
    Returns metrics for a random agent.
    """
    variedBaseSeeds = yamlConfig["varied_base_seeds"].copy()
    experimentConfig = setUpEvaluationConfig(yamlConfig, "random")
    averageRandomPerformance = []
    env = getEnv(yamlConfig)
    env.setup(yamlConfig)
    for rep in range(randomRepeats):
        if rep % randomRepeats // len(variedBaseSeeds) == 0 and variedBaseSeeds:
            BASE_SEED = variedBaseSeeds.pop()
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
    randomMetrics = getRandomMetrics(yamlConfig, randomRepeats=1000)
    bestTestSetPerformance = plotLearningCurves(
        yamlConfig=yamlConfig,
        randomMetrics=randomMetrics,
        rewardFunctions=rewards,
        agentType=agentType,
        sumTestTrainingPeriods=envDetails["sum_test_training_periods"],
    )
    tabulateBestTestSetPerformance(
        yamlConfig, bestTestSetPerformance, rewards, randomMetrics
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
        avRandReturn=randomMetrics["average_random_performance"],
        rewards=rewards,
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
                )
                wandb.finish()

        testMetricsAndGraphs(
            yamlConfig, rewards=REWARDS["Reward"], envDetails=env.datasetsAndDetails
        )
