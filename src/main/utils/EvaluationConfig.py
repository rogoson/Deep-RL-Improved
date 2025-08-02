def setUpEvaluationConfig(yamlConfig, stage):

    experimentConfig = yamlConfig["experiments"].get(stage, {})
    finalConfig = {
        "dataType": experimentConfig.get("dataType"),
        "benchmark": (experimentConfig.get("benchmark")),
        "comparisonStrat": experimentConfig.get("comparisonStrategy"),
        "useNoiseEval": experimentConfig.get("use_noise_eval"),
        "forLearningCurve": experimentConfig.get("for_learning_curve"),
        "baseline": yamlConfig.get("baseline"),
        "rl_strats": yamlConfig.get("rl_strats"),
        "sourceFolder": experimentConfig.get("sourceFolder", "main"),
    }

    return finalConfig
