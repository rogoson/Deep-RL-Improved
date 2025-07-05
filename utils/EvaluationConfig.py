import argparse


def setUpEvaluationConfig(yamlConfig, stage):
    parser = argparse.ArgumentParser(description="Set up Evaluation Function Call")

    parser.add_argument(
        "--datatype",
        type=str,
        choices=["validation", "testing"],
        help="Type of data used",
    )
    parser.add_argument("--save", action="store_true", help="Flag to save results")
    parser.add_argument(
        "--benchmark",
        type=lambda x: x.lower() == "true" if isinstance(x, str) else bool(x),
        help="Random runs",
    )
    parser.add_argument(
        "--compare", type=str, default=None, help="Optional comparison strategy"
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
        "datatype": args.datatype or experimentConfig.get("datatype"),
        "save": args.save or experimentConfig.get("save"),
        "benchmark": (
            args.benchmark
            if args.benchmark is not None
            else experimentConfig.get("benchmark")
        ),
        "compare": args.compare or experimentConfig.get("compare"),
        "use_noise_eval": args.use_noise_eval or experimentConfig.get("use_noise_eval"),
        "for_learning_curve": args.for_learning_curve
        or experimentConfig.get("for_learning_curve"),
        "baselines": args.baselines or yamlConfig.get("baseline"),
        "rl_strats": args.rl_strats or yamlConfig.get("rl_strats"),
    }

    return finalConfig
