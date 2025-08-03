def storeExperiences(
    agent=None,
    data=None,
    reward=None,
    nextData=None,
    done=None,
    strategy=None,
    action=None,
    probabilities=None,
    valuation=None,
    hiddenAndCellStates=None,
):
    """
    Store experiences in the agent's memory.
    """
    if strategy == "PPOLSTM":
        agent.store(
            state=data,
            action=action,
            probability=probabilities.squeeze(),
            valuation=valuation.squeeze(),
            reward=reward,
            done=done,
            hiddenStates=hiddenAndCellStates,
        )
    else:  # TD3
        agent.store(
            state=data,
            action=action,
            reward=reward,
            nextState=nextData,
            done=done,
            hiddenStates=hiddenAndCellStates,
        )


def hiddenStateReset(agent):
    """
    Resets the hidden states of the agent and feature extractor.
    This is called at the start of any training/testing episode
    to prevent temporoal leakage.
    """
    hAndCStates = dict()
    if agent.__class__.__name__ == "TD3Agent":
        hAndCStates["critic2"] = agent.critic2.initHidden(batchSize=1)
        hAndCStates["targetActor"] = agent.targetActor.initHidden(batchSize=1)
        hAndCStates["targetCritic"] = agent.targetCritic.initHidden(batchSize=1)
        hAndCStates["targetCritic2"] = agent.targetCritic2.initHidden(batchSize=1)
        hAndCStates["targetFeature"] = agent.targetFeatureExtractor.initHidden(
            batchSize=1
        )
    hAndCStates["actor"] = agent.actor.initHidden(batchSize=1)
    hAndCStates["critic"] = agent.critic.initHidden(batchSize=1)
    hAndCStates["feature"] = agent.featureExtractor.initHidden(batchSize=1)
    return hAndCStates
