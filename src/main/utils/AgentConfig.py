from main.agents.PPO import PPOAgent
from main.featureExtractors.LstmFeatureExtractor import LstmFeatureExtractor


def createAgentFromConfig(
    agentType,
    phase,
    yamlConfig,
    optionalHyperConfig=None,
    numberOfFeatures=None,  # better hope to God that this exists
    featureExtractor=None,
):
    """
    Builds configuration and returns an RL agent instance using config-driven parameters.
    Supports multiple agent types like 'ppo', 'td3', etc.
    """
    if agentType not in yamlConfig["agent"]:
        raise ValueError(f"Agent type '{agentType}' not found in configuration.")

    agentCfg = yamlConfig["agent"][agentType]

    # Common config base
    baseConfig = {
        "gamma": yamlConfig["agent"]["gamma"],
        "actions_n": int((numberOfFeatures - 2) / 7) + 1,  # BRAVE
        "batch_size": agentCfg.get("batch_size", 64),
        "learning_rate": yamlConfig["agent"]["learning_rate"],
        "reward_function": yamlConfig["agent"]["reward_function"],
        "number_of_features": numberOfFeatures,
        "time_window": yamlConfig.get("time_window", 0),
    }

    # Agent-specific config logic
    if agentType == "ppo":
        baseConfig.update(
            {
                "gae_lambda": agentCfg["gae_lambda"],
                "clip_param": agentCfg["clip_param"],
                "fc1_n": agentCfg["fc1_n"],
                "fc2_n": agentCfg["fc2_n"],
                "lstm_output_size": agentCfg["feature_output_size"],
                "epochs": agentCfg["epochs"],
                "entropy_coef": agentCfg["entropy_coef"],
                "actor_noise": agentCfg.get("actor_noise", 0),
                "norm_advantages": agentCfg.get("norm_advantages", False),
                "use_entropy": agentCfg.get("use_entropy", False),
                "use_dirichlet": agentCfg.get("use_dirichlet", True),
                "log_concentration": agentCfg.get("log_concentration", False),
                "lstm_hidden_size": {
                    "actor": agentCfg["actor_critic_hidden_state_size"],
                    "critic": agentCfg["actor_critic_hidden_state_size"],
                    "feature": yamlConfig["feature_extractors"]["lstm"][
                        "default_hidden_size"
                    ],
                },
            }
        )

    elif agentType == "td3":
        pass
        # baseConfig.update({
        #     "policy_noise": agentCfg["policy_noise"],
        #     "noise_clip": agentCfg["noise_clip"],
        #     "tau": agentCfg["tau"],
        # })

    else:
        raise ValueError(f"Agent type '{agentType}' is not yet supported.")

    # Phase-specific overrides
    if phase == "data_normalisation":
        baseConfig.update({"phase": phase, "group": "Data Normalisation"})

    elif phase == "noise_testing":
        baseConfig.update({"phase": phase, "group": "Noise Variation"})

    elif phase == "hyperparameter_tuning":
        baseConfig.update(
            {
                "learning_rate": (
                    optionalHyperConfig.get(
                        "learning_rate", baseConfig["learning_rate"]
                    )
                    if optionalHyperConfig
                    else baseConfig["learning_rate"]
                ),
                "lstm_output_size": (
                    optionalHyperConfig.get(
                        "lstm_output_size", baseConfig.get("lstm_output_size", 128)
                    )
                    if optionalHyperConfig
                    else baseConfig.get("lstm_output_size", 128)
                ),
                "phase": phase,
                "group": "Hyperparameter Tuning",
            }
        )

    elif phase == "reward_testing":
        reward_function = (
            optionalHyperConfig.get("reward_function", baseConfig["reward_function"])
            if optionalHyperConfig
            else baseConfig["reward_function"]
        )
        baseConfig.update(
            {
                "phase": phase,
                "reward_function": reward_function,
                "risk_aversion": (
                    float(reward_function.split("_")[1])
                    if "CVaR" in reward_function
                    else 0
                ),
                "group": "Reward Function Variation",
            }
        )

    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Feature extractor (if required)
    if featureExtractor is None and agentType == "ppo":
        featureExtractor = LstmFeatureExtractor(
            baseConfig["number_of_features"],
            lstmHiddenSize=baseConfig["lstm_hidden_size"]["feature"],
            lstmOutputSize=baseConfig.get("lstm_output_size", 128),  # Default
        )

    # === AGENT CREATION ===
    if agentType == "ppo":
        return {
            "agent": PPOAgent(
                state_n=baseConfig.get("lstm_output_size", 128),
                actions_n=baseConfig.get("actions_n", 1),
                alpha=baseConfig["learning_rate"],
                policyClip=baseConfig.get("clip_param", 0.2),
                gamma=baseConfig.get("gamma", 0.99),
                lstmHiddenSizeDictionary=baseConfig.get(
                    "lstm_hidden_size", None
                ),  # goes boom boom if missing
                actor_noise=baseConfig.get("actor_noise", 0),
                batch_size=baseConfig["batch_size"],
                fc1_n=baseConfig.get("fc1_n", 128),
                fc2_n=baseConfig.get("fc2_n", 128),
                gaeLambda=baseConfig.get("gae_lambda", 0.98),
                epochs=baseConfig.get("epochs", 1),
                riskAversion=baseConfig.get("risk_aversion", 0),
                featureExtractor=featureExtractor,
                maxSize=baseConfig["batch_size"],
                nonFeatureStateDim=(
                    baseConfig.get("time_window", 0),
                    baseConfig.get("number_of_features", 0),
                ),
                entropyCoefficient=baseConfig.get("entropy_coef", 0.01),
                rewardFunction=baseConfig.get(
                    "reward_function", "Standard Logarithmic Returns"
                ),
                normAdvantages=baseConfig.get("norm_advantages", False),
                useEntropy=baseConfig.get("use_entropy", False),
                useDirichlet=baseConfig.get("use_dirichlet", True),
                log_concentration_heatmap=baseConfig.get("log_concentration", False),
                experimentState=phase,
            ),
            "agentConfig": baseConfig,
        }

    elif agentType == "td3":
        pass
        # return {"agent": TD3Agent(  # Hypothetical example
        #     state_dim=(baseConfig["time_window"], baseConfig["number_of_features"]),
        #     action_dim=baseConfig["actions_n"],
        #     gamma=baseConfig["gamma"],
        #     alpha=baseConfig["learning_rate"],
        #     tau=baseConfig["tau"],
        #     policy_noise=baseConfig["policy_noise"],
        #     noise_clip=baseConfig["noise_clip"],
        #     policy_delay=baseConfig["policy_delay"],
        #     batch_size=baseConfig["batch_size"],
        # experimentState=phase,
        #     # Include other TD3-specific args
        # ), "agentConfig": baseConfig}

    else:
        raise NotImplementedError(
            f"Agent type '{agentType}' instantiation not implemented."
        )
