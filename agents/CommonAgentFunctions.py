def storeExperiences(
    agent, data, reward, done, strategy, action, prob, val, hiddenAndCStates=None
):
    """
    Store experiences in the agent's memory.
    """
    if strategy == "PPOLSTM":
        agent.store(
            data, action, prob.squeeze(), val.squeeze(), reward, done, hiddenAndCStates
        )


def hiddenStateReset(agent):
    """
    Resets the hidden states of the agent and feature extractor.
    This is called at the start of any training/testing episode
    to prevent temporoal leakage.
    """
    hAndCStates = dict()
    hAndCStates["actor"] = agent.actor.initHidden(batchSize=1)
    hAndCStates["critic"] = agent.critic.initHidden(batchSize=1)
    hAndCStates["feature"] = agent.featureExtractor.initHidden(batchSize=1)
    return hAndCStates
