import argparse


def setUpEvaluationConfig(yamlConfig, stage):
    parser = argparse.ArgumentParser(description="Set up Evaluation Function Call")

    parser.add_argument(
        "--dataType",
        type=str,
        choices=["validation", "testing"],
        help="Type of data used",
    )
    parser.add_argument(
        "--benchmark",
        type=lambda x: x.lower() == "true" if isinstance(x, str) else bool(x),
        help="Random runs",
    )
    parser.add_argument(
        "--comparisonStrat",
        type=str,
        default=None,
        help="Optional comparison strategy",
    )
    parser.add_argument(
        "--use_noise_eval", action="store_true", help="Enable noise for evaluation"
    )
    parser.add_argument(
        "--for_learning_curve",
        type=bool,
        default=False,
        help="Generating learning curve data?",
    )
    parser.add_argument(
        "--baseline", help="Baselines to compare against", nargs="*", default=None
    )
    parser.add_argument(
        "--rl_strats", help="RL strategies to compare against", nargs="*", default=None
    )

    args = parser.parse_args()

    experimentConfig = yamlConfig["experiments"].get(stage, {})
    finalConfig = {
        "dataType": args.dataType or experimentConfig.get("dataType"),
        "benchmark": (
            args.benchmark
            if args.benchmark is not None
            else experimentConfig.get("benchmark")
        ),
        "comparisonStrat": args.comparisonStrat
        or experimentConfig.get("comparisonStrategy"),
        "useNoiseEval": args.use_noise_eval or experimentConfig.get("use_noise_eval"),
        "forLearningCurve": args.for_learning_curve
        or experimentConfig.get("for_learning_curve"),
        "baseline": args.baseline or yamlConfig.get("baseline"),
        "rl_strats": args.rl_strats or yamlConfig.get("rl_strats"),
    }

    return finalConfig
