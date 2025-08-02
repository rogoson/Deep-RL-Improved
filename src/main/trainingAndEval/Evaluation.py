from main.agents.CommonAgentFunctions import hiddenStateReset
from main.utils.GeneralUtils import getFileWritingLocation
import matplotlib.pyplot as plt
import numpy as np
import os

NON_RL_COMPARISON_STRATEGIES = [
    "SSE 50 Buy-and-Hold",
    "SENSEX Buy-and-Hold",
    "FTSE 100 Buy-and-Hold",
    "S&P 500 Buy-and-Hold",
    "ALL Buy-and-Hold",
]

LOG_OBSERVATIONS = False
LOG_INPUT_DATA = False
LOG_ANY = LOG_OBSERVATIONS or LOG_INPUT_DATA


def logDetails(LOG_DETAILS):
    LOG_INPUT_DATA = LOG_DETAILS["inputData"][0]
    LOG_OBSERVATIONS = LOG_DETAILS["observations"][0]

    dataOverTime = LOG_DETAILS["inputData"][1]
    observationsOverTime = LOG_DETAILS["observations"][1]

    if LOG_INPUT_DATA:
        plt.figure(figsize=(12, 8))  # Bigger figure for clarity
        for dim in range(dataOverTime[0].shape[0]):
            plt.plot(
                np.arange(len(dataOverTime)), np.array(dataOverTime)[:, dim], alpha=0.5
            )  # Adjust transparency to make it readable

        plt.xlabel("Time")
        plt.ylabel("Values per Index")
        plt.title("Data Over Time (using data index 0)")
        plt.grid(True, linestyle="--", alpha=0.5)  # Add a subtle grid for readability
        plt.show()
    if LOG_OBSERVATIONS:
        plt.figure(figsize=(12, 8))  # Bigger figure for clarity
        for dim in range(observationsOverTime[0].shape[0]):
            plt.plot(
                np.arange(len(observationsOverTime)),
                np.array(observationsOverTime)[:, dim],
                alpha=0.5,
            )  # Adjust transparency to make it readable

        plt.xlabel("Time")
        plt.ylabel("Values per Index")
        plt.title("Observations Vectors Over Time")
        plt.grid(True, linestyle="--", alpha=0.5)  # Add a subtle grid for readability
        plt.show()


def evaluateAgent(
    agent,
    env,
    num,
    conf=None,
    dataType="validation",
    forLearningCurve=False,
    benchmark=False,
    epoch=0,
    comparisonStrat=None,
    useNoiseEval=True,
    stage=None,
    baseline=None,
    rl_strats=None,
    sourceFolder=None,
):
    """
    Docstring would be helpful.
    """
    toRun = rl_strats
    if benchmark:
        """
        slightly misleading. If using the random agent as a benchmark, the random agent is seeded by that
        repetition to ensure that many random seeds are used for it
        """
        np.random.seed(env.baseSeed + epoch)
        toRun = baseline
    if comparisonStrat is not None:
        """
        An optional comparison strategy. This is used to compare the agent's performance against an index.
        """
        strategy = comparisonStrat[0]
        strategyVector = comparisonStrat[1]
        toRun = [strategy]

    for strategy in toRun:

        env.reset(evalType=dataType)
        env.setData(dataType=dataType, useNoiseEval=useNoiseEval, epoch=epoch)

        done = False
        if strategy == "PPOLSTM":
            """
            Reset hidden and cell states of the agent and feature extractor.
            """
            hiddenAndCellStates = hiddenStateReset(agent)
        observationsOverTime = []
        dataOverTime = []
        while not done:
            if strategy not in NON_RL_COMPARISON_STRATEGIES:
                if not env.getIsReady():
                    """
                    If the strategy being tested is random/PPO, warm up the environment (to ensure that they both have the same starting point)
                    """
                    env.warmUp(observeReward=False)

            observation = None
            if strategy in rl_strats:
                data = env.getData()  # Retrieve data
                dataOverTime.append(data.squeeze(0)[0].detach().cpu().numpy())
                observation, hiddenAndCellStates["feature"] = (
                    agent.featureExtractor.forward(data, hiddenAndCellStates["feature"])
                )
                if LOG_OBSERVATIONS:
                    observationsOverTime.append(observation.detach().cpu().numpy())

            if strategy == "RANDOM":
                """Random agent samples actions from a Dirichlet distribution - The same as that used in the PPO agent for consistency."""
                action = np.random.dirichlet(np.ones(env.numberOfAssets + 1))
            elif strategy in NON_RL_COMPARISON_STRATEGIES:
                action = strategyVector
            else:
                if strategy == "PPOLSTM":
                    action, _, __, actorHidden, criticHidden = agent.select_action(
                        observation,
                        hiddenAndCellStates,
                        sampling=False,
                        returnHidden=True,
                    )
                    hiddenAndCellStates["actor"] = (
                        actorHidden  # #update and cell states of actor
                    )
                    hiddenAndCellStates["critic"] = (
                        criticHidden  # update and cell states of critic
                    )
            next, reward, done, _, info = env.step(
                action, returnNextObs=False, observeReward=False
            )  # the reward is not observed during evaluation, since the agent does not learn from the data, further, nextobs is not required since no GAE

        env.rendering = False

        if strategy in rl_strats:
            generateAnimation = agent.save(env.PORTFOLIO_VALUES[-1] / env.startCash)
            if generateAnimation:
                env.generateAnimation(
                    agentType=agent.__class__.__name__, stage=agent.experimentState
                )

        if strategy in rl_strats:
            if LOG_ANY:
                # Dictionary of boolean flags to data to log
                LOG_DETAILS = {
                    "observations": [LOG_OBSERVATIONS, observationsOverTime],
                    "inputData": [LOG_INPUT_DATA, dataOverTime],
                }
                logDetails(LOG_DETAILS)

        """
        The below is a lot of 'saving' code, for saving models and portfolio trajectories where necessary
        """

        if forLearningCurve:
            portFolder = (
                getFileWritingLocation(sourceFolder)
                + f"/portfolios/{dataType}/forLearningCurve{env.baseSeed}/"
            )
            if not os.path.exists(portFolder):
                os.makedirs(portFolder)

            filePath = f"{portFolder}{(conf.split('|')[0]).strip()}_{num}.txt"
            np.savetxt(filePath, env.PORTFOLIO_VALUES, fmt="%f")

        if (
            benchmark
            or strategy in NON_RL_COMPARISON_STRATEGIES
            or (not useNoiseEval and strategy in rl_strats)
        ):
            # Sometimes it is necessary (when not saving models) to simply return the portfolio values
            return env.PORTFOLIO_VALUES

    return env.PORTFOLIO_VALUES  # VERY hacky - returns portfolio values for rl strat
