import wandb
import time
from agents.CommonAgentFunctions import hiddenStateReset, storeExperiences
from utils.AgentConfig import createAgentFromConfig
from Evaluation import evaluateAgent
from utils.EvaluationConfig import setUpEvaluationConfig
from experiments.InitialisationHelpers import initialiseWandb


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
    agent, agentConfig = createAgentFromConfig(
        agentType,
        phase=stage,
        yamlConfig=yamlConfig,
        numberOfFeatures=envAgentInformation["numberOfFeatures"],
    )
    initialiseWandb(yamlConfig, agent, agentConfig)
    env.setRewardFunction(agent.rewardFunction)

    experimentConfig = setUpEvaluationConfig(yamlConfig, stage)
    strategy = yamlConfig["agent"][agentType]["strategy"]

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
        for episode in range(env.datasetsAndDetails["training_windows"]):
            print("Episode:", episode)
            env.reset()
            hiddenAndCellStates = hiddenStateReset(agent)
            done = False
            while not done:
                if not env.getIsReady():
                    env.warmUp(env, agent.rewardFunction)
                    continue
                observation = None
                data = env.getData()
                if strategy == "PPOLSTM":
                    prevHiddenAndCellStates = (
                        hiddenAndCellStates.copy()
                    )  # save previous hidden and cell states (for storing)
                observation, hiddenAndCellStates["feature"] = (
                    agent.featureExtractor.forward(data, hiddenAndCellStates["feature"])
                )
                probabilities, valuation = None, None
                if strategy == "PPOLSTM":
                    action, probabilities, valuation, actorHidden, criticHidden = (
                        agent.select_action(
                            observation, hiddenAndCellStates, returnHidden=True
                        )
                    )
                    hiddenAndCellStates["actor"] = actorHidden  # returned from agent
                    hiddenAndCellStates["critic"] = criticHidden
                next, reward, done, _, info = env.step(action)
                totalReward += reward
                totalTimesteps += 1
                if strategy in yamlConfig["rl_strats"]:
                    storeExperiences(
                        agent,
                        data,
                        reward,
                        done,
                        strategy,
                        action,
                        probabilities,
                        valuation,
                        prevHiddenAndCellStates if strategy == "PPOLSTM" else None,
                    )
                    if (
                        agent.memory.ptr % agent.memory.maxSize == 0
                    ):  # Train when the batch is full - following Zou et al. (2024) with their single epoch training loop.
                        agent.train(
                            next, hiddenAndCellStates
                        )  # required for proper GAE
                    if experimentConfig["datatype"] == "testing" and (
                        totalTimesteps % yamlConfig["test"]["learning_curve_frequency"]
                        == 0
                    ):
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
                if done:
                    numberRun += 1
                    print("Episode Reward:", totalReward - previousReward)
                    wandb.log({"total_reward": totalReward}, commit=False)
                    previousReward = totalReward
                    if experimentConfig[
                        "use_noise_eval"
                    ]:  # if not doing noise testing, only evaluate once per epoch
                        if numberRun % env.datasetsAndDetails["training_windows"] == 0:
                            evaluateAgent(
                                agent=agent,
                                env=env,
                                num=numberRun,
                                conf=conf,
                                **experimentConfig,
                            )
                    else:  # if doing noise testing, evaluate at he end of each training ep
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
    return trainingMetrics if not experimentConfig["use_noise_eval"] else None
