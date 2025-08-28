import wandb
import time
import copy
import numpy as np
from main.agents.CommonAgentFunctions import hiddenStateReset, storeExperiences
from main.agents.AgentConfig import createAgentFromConfig
from .Evaluation import evaluateAgent
from main.utils.EvaluationConfig import setUpEvaluationConfig
from main.experiments.InitialisationHelpers import initialiseWandb


def trainingLoop(
    yamlConfig,
    env,
    agentType="ppo",
    conf=None,
    stage=None,
    optionalHyperConfig=None,
):
    """
    Main training loop for the agent.
    :param yamlConfig: Configuration dictionary loaded from YAML file.
    :param agentType: Type of agent to be used (default is "ppo").
    """
    # Environment Setup and Initialisation
    envAgentInformation = env.setup(yamlConfig)

    # Agent Details and Initialisation
    agentAndAgentConfig = createAgentFromConfig(
        agentType,
        phase=stage,
        yamlConfig=yamlConfig,
        numberOfFeatures=envAgentInformation["numberOfFeatures"],
        optionalHyperConfig=optionalHyperConfig,
    )
    agent = agentAndAgentConfig["agent"]
    agentConfig = agentAndAgentConfig["agentConfig"]

    initialiseWandb(yamlConfig, agent, agentConfig)
    env.setRewardFunction(agent.rewardFunction)

    strategy = yamlConfig["agent"][agentType]["strategy"]
    experimentConfig = setUpEvaluationConfig(
        yamlConfig, stage, currentStrategy=strategy
    )

    trainingMetrics = {
        "epoch_reward": [],  # to store validtion set peformances when noise testing
        "validation_performances": [],  # to store total rewards for each training episode
    }

    numberRun = 0
    totalTimesteps = 0
    startTime = time.time()  # just to time epochs
    for epoch in range(yamlConfig["epochs"]):
        print("Epoch:", epoch)
        if epoch > 0:
            print(f"{epoch} Epochs takes: {(time.time() - startTime):.2f} seconds")
        """
        Much of the rest of the code follows a similar structure to the evaluation function, except for training.
        """
        previousReward = 0
        totalReward = 0

        episodeOrder = np.random.permutation(  # randomise episode order, to limit temporal correlation for td3
            env.datasetsAndDetails[
                (
                    "training_windows"
                    if stage != "reward_testing"
                    else "test_training_windows"
                )
            ]
        )

        for episode in episodeOrder:
            print("Episode:", np.where(episodeOrder == episode)[0][0])
            env.reset(  # gotta have so much faith in this
                evalType="validation" if stage != "reward_testing" else "testing",
                pushWindow=True,
                episode=episode,
                epoch=epoch,
            )
            hiddenAndCellStates = hiddenStateReset(agent)
            done = False
            while not done:
                if not env.getIsReady():
                    env.warmUp()
                    continue
                observation = None
                data = env.getData()
                if strategy in yamlConfig["rl_strats"]:
                    prevHiddenAndCellStates = (
                        hiddenAndCellStates.copy()
                    )  # save previous hidden and cell states (for storing)
                observation, hiddenAndCellStates["feature"] = (
                    agent.featureExtractor.forward(
                        data,
                        (
                            hiddenAndCellStates["feature"]
                            if not agent.hasCNNFeature
                            else None
                        ),
                    )
                )
                if strategy == "TD3":
                    if not agent.hasCNNFeature:
                        _, hiddenAndCellStates["targetFeature"] = (
                            agent.targetFeatureExtractor.forward(
                                data,
                                (hiddenAndCellStates["feature"]),
                            )
                        )

                if strategy in yamlConfig["rl_strats"]:
                    action = None
                    probabilities, valuation = None, None
                    if strategy == "PPOLSTM":
                        action, probabilities, valuation, actorHidden, criticHidden = (
                            agent.select_action(
                                observation, hiddenAndCellStates, returnHidden=True
                            )
                        )

                        hiddenAndCellStates["critic"] = criticHidden
                    elif strategy == "TD3":
                        (
                            action,
                            actorHidden,
                            criticHidden,
                            critic2Hidden,
                            targetActorHidden,
                            targetCriticHidden,
                            targetCritic2Hidden,
                        ) = agent.select_action(
                            observation, hiddenAndCellStates, returnHidden=True
                        )

                        hiddenAndCellStates["critic"] = criticHidden
                        hiddenAndCellStates["critic2"] = critic2Hidden
                        hiddenAndCellStates["targetActor"] = targetActorHidden
                        hiddenAndCellStates["targetCritic"] = targetCriticHidden
                        hiddenAndCellStates["targetCritic2"] = targetCritic2Hidden
                    hiddenAndCellStates["actor"] = actorHidden

                next, reward, done, _, info = env.step(action)
                totalReward += reward
                totalTimesteps += 1
                if strategy in yamlConfig["rl_strats"]:
                    storeExperiences(
                        agent=agent,
                        data=data,  # raw observation matrix
                        reward=reward,
                        nextData=next,  # needs to be gracefully handled
                        done=done,
                        strategy=strategy,
                        action=action,
                        probabilities=probabilities,
                        valuation=valuation,
                        hiddenAndCellStates=(prevHiddenAndCellStates),
                    )
                    if strategy == "PPOLSTM":
                        if agent.memory.ptr % agent.memory.maxSize == 0:
                            agent.learn(
                                next, hiddenAndCellStates
                            )  # required for proper GAE for ppo
                    elif strategy == "TD3":
                        if agent.memory.ptr >= agent.batchSize:
                            agent.learn()
                    if experimentConfig["dataType"] == "testing" and (
                        totalTimesteps % yamlConfig["test"]["learning_curve_frequency"]
                        == 0
                    ):
                        temporaryEnvState = copy.deepcopy(
                            env.__dict__
                        )  # bad design - pausing unfinished run
                        learningCurveValues = evaluateAgent(
                            agent=agent,
                            env=env,
                            num=numberRun,
                            conf=conf,
                            **experimentConfig,
                        )
                        wandb.log(
                            {
                                "learning_curve (test)": learningCurveValues[-1]
                                / env.startCash
                            },
                            commit=False,
                        )
                        env.__dict__.clear()  # restoring unfinished run
                        env.__dict__.update(temporaryEnvState)
                if done:
                    numberRun += 1
                    print("Episode Reward:", totalReward - previousReward)
                    wandb.log({"total_reward": totalReward}, commit=False)
                    previousReward = totalReward
                    # if doing noise testing, evaluate at he end of each training ep
                    if not experimentConfig[
                        "forLearningCurve"
                    ]:  # handled inside training loop - not for this exp
                        portTrajectory = evaluateAgent(
                            agent=agent,
                            env=env,
                            num=numberRun,
                            conf=conf,
                            **experimentConfig,
                        )
                        trainingMetrics["validation_performances"].append(
                            portTrajectory[-1] / env.startCash
                        )  # store the last value of the portfolio trajectory
                        wandb.log(
                            {
                                "evaluation_performances": trainingMetrics[
                                    "validation_performances"
                                ][-1]
                            },
                            commit=False,
                        )
            trainingMetrics["epoch_reward"].append(totalReward)
    return trainingMetrics
