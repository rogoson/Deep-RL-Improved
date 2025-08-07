from main.utils.GeneralUtils import seed, getFileWritingLocation
from main.trainingAndEval.Training import trainingLoop
from .NonTestExperimentsPlotting import (
    plotNormalisationExpPerformance,
    runNoiseComparison,
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


def normalisationEffectExperiment(
    yamlConfig, agentType="ppo", phase="data_normalisation"
):
    """
    Test the effect of normalisation on the agent's performance.
    """
    portfolioValues = dict()
    for isNormalised in [True, False]:
        normFolder = (
            getFileWritingLocation(yamlConfig, agentType)
            + f"/portfolios/{'Normalisation' if isNormalised else 'NonNormalisation'}/"
        )
        for s in yamlConfig["varied_base_seeds"]:
            BASE_SEED = s
            seed(BASE_SEED)  # Seed agent initialization and updates

            yamlConfig["perturbation_noise"] = 0
            yamlConfig["normalise_data"] = isNormalised
            yamlConfig["env"]["base_seed"] = s

            print("*" * 50)
            print("Testing Seed: ", s)
            env = getEnv(yamlConfig)
            portfolioValues[s] = trainingLoop(yamlConfig, env, agentType, stage=phase)
            desiredFolder = f"{normFolder}{s}/"
            if not os.path.exists(desiredFolder):
                os.makedirs(desiredFolder)
            np.savetxt(
                f"{desiredFolder}validationPerformances.txt",
                portfolioValues[s]["validation_performances"],
                fmt="%f",
            )
            np.savetxt(
                f"{desiredFolder}trainingRewards.txt",
                portfolioValues[s]["epoch_reward"],
                fmt="%f",
            )

            print("*" * 50)
            wandb.finish()
    print("Normalisation Effect Experiment Completed")
    fileWritingLocation = getFileWritingLocation(yamlConfig, agentType)

    base = Path(fileWritingLocation)
    (base / "plots").mkdir(parents=True, exist_ok=True)
    (base / "portfolios").mkdir(parents=True, exist_ok=True)

    plotNormalisationExpPerformance(
        base / "portfolios/Normalisation",
        title="With Normalisation (Rolling Z-Score)",
        saveFile=base / "plots/NormalisationEffect.png",
        agentType=agentType,
    )

    plotNormalisationExpPerformance(
        base / "portfolios/NonNormalisation",
        title="Without Normalisation",
        saveFile=base / "plots/NonNormalisationEffect.png",
        agentType=agentType,
    )


def hyperparameterTuning(yamlConfig, agentType="ppo", phase="hyperparameter_tuning"):
    """
    Runs the hyperparameter sweep by sequentially activating one test type at a time.
    """
    TESTING = {
        "FEATURE OUTPUT SIZE": False,
        "LEARNING RATE": False,
    }
    for key in list(TESTING.keys()):
        print("=" * 50)
        TESTING[key] = True

        sweepParams = {
            "FEATURE OUTPUT SIZE": {
                "values": yamlConfig["hyperparameters"]["feature_output_sizes"],
                "overrides": {"lstm_output_size": None},
            },
            "LEARNING RATE": {
                "values": yamlConfig["hyperparameters"]["learning_rates"],
                "overrides": {"learning_rate": None},
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
                        env = getEnv(yamlConfig)
                        trainingLoop(
                            yamlConfig,
                            env,
                            agentType,
                            stage=phase,
                            conf=f"{testType.lower().title()} - {value} | Strategy-{agentType}",
                            optionalHyperConfig=overrides,
                        )
                        wandb.finish()
                    break  # Run only one active test type per sweep
        TESTING[key] = False
        print("=" * 50)


def getRandomMetrics(yamlConfig, dataType="validation", randomRepeats=1000):
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
    randomMetrics = getRandomMetrics(yamlConfig, dataType="testing", randomRepeats=1000)
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
        experimentConfig["comparisonStrat"] = strat
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
            "CVaR_0.25",
            "CVaR_0.5",
            "CVaR_1.0",
            "CVaR_1.5",
            "CVaR_2.0",
        ]
    }
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
