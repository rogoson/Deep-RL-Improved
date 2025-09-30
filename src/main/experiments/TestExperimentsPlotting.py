from itertools import cycle
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .MetricComputations import scoreFormula, maxDrawdown
from main.utils.TabulationUtils import format_latex_table
from main.utils.GeneralUtils import getFileWritingLocation


def plotLearningCurves(
    yamlConfig,
    rewardFunctions,
    randomMetrics,
    sumTestTrainingPeriods,
    agentType="ppo",
    WINDOW=5,
    FIG_SIZE=(12, 8),
):
    bestTestSetPerformance = dict()
    for seed in yamlConfig["varied_base_seeds"]:
        cumulativeReturnsData = {}
        scoresData = {}
        timeSteps = []

        for rewardFunc in rewardFunctions:
            file = f"{getFileWritingLocation(yamlConfig, agentType=agentType)}/portfolios/testing/forLearningCurve{seed}/Reward Function-{rewardFunc}_"
            cumReturns = []
            scores = []
            for i in range(1, yamlConfig["test"]["learning_curve_frequency"]):
                key = f"{file}{i}.txt"
                values = None
                try:
                    values = np.loadtxt(key)
                except FileNotFoundError:
                    continue
                metrics = scoreFormula(
                    values, randomMetrics["average_random_return"], yamlConfig
                )
                cumReturns.append(metrics["Cumulative \nReturn (%)"])
                scores.append(metrics["Score"])
                if len(timeSteps) < yamlConfig["test"]["learning_curve_frequency"] - 1:
                    timeSteps.append(
                        i
                        / yamlConfig["test"]["learning_curve_frequency"]
                        * sumTestTrainingPeriods
                    )
            cumulativeReturnsData[rewardFunc] = cumReturns
            scoresData[rewardFunc] = scores
            bestPercentThrough = np.argmax(scores)
            bestTestSetPerformance[(rewardFunc, seed)] = [
                max(cumReturns),
                scores[bestPercentThrough],
                int(timeSteps[bestPercentThrough]),
            ]
            bestTestSetPerformance[(rewardFunc, seed)].append(
                np.loadtxt(f"{file}{bestPercentThrough + 1}.txt")
            )

        # Plot cumulative returns for this seed
        plt.figure(figsize=FIG_SIZE)
        for rewardFunc, cumReturns in cumulativeReturnsData.items():
            properName = rewardFunc.split("_")
            if len(properName) > 1:
                if "CVaR" in properName[0]:
                    plottedName = f"CVaR ($\\zeta={properName[1]}$)"
                elif "Differential Sharpe Ratio" in properName[0]:
                    plottedName = f"Differential Sharpe Ratio ($\\eta={properName[1]}$)"
                else:
                    plottedName = f"{properName[0]} ({properName[1]})"
            else:
                plottedName = properName[0]
            smoothedReturns = np.convolve(
                cumReturns, np.ones(WINDOW) / WINDOW, mode="valid"
            )
            numTimeSteps = len(np.append(cumReturns[: WINDOW - 1], smoothedReturns))
            plt.plot(
                timeSteps[:numTimeSteps],
                np.append(cumReturns[: WINDOW - 1], smoothedReturns),
                label=plottedName,
                linewidth=2,
            )
        plt.axhline(
            y=randomMetrics["average_random_return"] * 100,
            color="grey",
            linestyle=":",
            label="Random (Baseline)",
            linewidth=1.5,
        )
        fileLoc = getFileWritingLocation(yamlConfig, agentType=agentType)
        plots = fileLoc + "/plots/"
        os.makedirs(plots, exist_ok=True)
        plt.xlabel("Training Timesteps Elapsed")
        plt.ylabel("Cumulative Returns (%)")
        plt.title(f"Learning Curve (Returns) – Seed {seed}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plots}/Cumulative_Returns_Seed{seed}")
        plt.close()

        # Plot scores for this seed
        plt.figure(figsize=FIG_SIZE)
        for rewardFunc, scores in scoresData.items():
            properName = rewardFunc.split("_")
            if len(properName) > 1:
                if "CVaR" in properName[0]:
                    plottedName = f"CVaR ($\\zeta={properName[1]}$)"
                elif "Differential Sharpe Ratio" in properName[0]:
                    plottedName = f"Differential Sharpe Ratio ($\\eta={properName[1]}$)"
                else:
                    plottedName = f"{properName[0]} ({properName[1]})"
            else:
                plottedName = properName[0]
            smoothedScores = np.convolve(scores, np.ones(WINDOW) / WINDOW, mode="valid")
            numTimeSteps = len(np.append(scores[: WINDOW - 1], smoothedScores))
            plt.plot(
                timeSteps[:numTimeSteps],
                np.append(scores[: WINDOW - 1], smoothedScores),
                label=plottedName,
                linewidth=2,
            )
        plt.axhline(
            y=0, color="grey", linestyle=":", label="Random (Baseline)", linewidth=1.5
        )
        plt.xlabel("Training Timesteps Elapsed")
        plt.ylabel("Scores")
        plt.title(f"Learning Curve (Scores) – Seed {seed}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plots}/Scores_Seed{seed}")
        plt.close()
    return bestTestSetPerformance


def tabulateBestTestSetPerformance(
    yamlConfig, bestTestSetPerformance, rewardFunctions, randomMetrics
):
    # Tabulate best agents for this seed
    for seed in yamlConfig["varied_base_seeds"]:
        table = pd.DataFrame(
            columns=[
                "Reward Function",
                "Cumulative Return",
                "Maximum Drawdown",
                "Sharpe Ratio",
                "Score",
                "Training Timesteps Elapsed",
            ]
        )

        for rewardFunc in rewardFunctions:
            properName = rewardFunc.split("_")
            tabulatedName = f"{properName[0] + (f' = {properName[1]}' if len(properName) > 1 else '')}"

            metrics = scoreFormula(
                bestTestSetPerformance[(rewardFunc, seed)][-1],
                randomMetrics["average_random_return"],
                yamlConfig=yamlConfig,
            )

            table.loc[len(table)] = [
                tabulatedName,
                metrics["Cumulative \nReturn (%)"],
                metrics["Maximum \nDrawdown (%)"],
                metrics["Sharpe Ratio"],
                metrics["Score"],
                bestTestSetPerformance[(rewardFunc, seed)][2],
            ]

        tablePath = getFileWritingLocation(yamlConfig) + "/tables/"
        num_cols = [c for c in table.columns if c != "Reward Function"]
        table[num_cols] = table[num_cols].round(4)
        os.makedirs(tablePath, exist_ok=True)
        print(format_latex_table(table))
        with open(tablePath + f"BestTestSet{seed}.tex", "w") as f:
            f.write(table.to_latex())


def bestPerformancesAndStandardDeviations(
    yamlConfig, bestTestSetPerformance, averPerformance, rewards, agentType="ppo"
):
    for rewardFunc in rewards:
        allTrajectories = []
        for seed in yamlConfig["varied_base_seeds"]:
            traj = (
                np.array(bestTestSetPerformance[(rewardFunc, seed)][-1])
                / yamlConfig["env"]["start_cash"]
                * 100
                - 100
            )
            allTrajectories.append(traj)

        # Convert to array for mean/std computation
        allTrajectories = np.array(allTrajectories)
        meanTrajectory = np.mean(allTrajectories, axis=0)
        std_traj = np.std(allTrajectories, axis=0)
        timesteps = np.arange(len(meanTrajectory))

        # Format name for legend and title
        properName = rewardFunc.split("_")
        if len(properName) > 1:
            if "CVaR" in properName[0]:
                plottedName = f"CVaR ($\\zeta={properName[1]}$)"
            elif "Differential Sharpe Ratio" in properName[0]:
                plottedName = f"Differential Sharpe Ratio ($\\eta={properName[1]}$)"
            else:
                plottedName = f"{properName[0]} ({properName[1]})"
        else:
            plottedName = properName[0]

        # Plot
        plt.figure(figsize=(10, 6))
        for traj in allTrajectories:
            plt.plot(
                timesteps, traj, color="green", alpha=0.3, linewidth=1
            )  # Faint individual runs
        plt.plot(
            timesteps,
            meanTrajectory,
            color="blue",
            label=f"{plottedName} (Mean)",
            linewidth=2.5,
        )
        plt.plot(
            np.array(averPerformance) / yamlConfig["env"]["start_cash"] * 100 - 100,
            label="Random (Baseline)",
            color="grey",
        )
        plt.fill_between(
            timesteps,
            meanTrajectory - std_traj,
            meanTrajectory + std_traj,
            color="blue",
            alpha=0.2,
        )
        # Clean filename string: replace space, dot, brackets, equal signs

        safe_filename = (
            rewardFunc.replace(" ", "_")
            .replace("=", "")
            .replace("(", "")
            .replace(")", "")
            .replace(".", "_")
        )
        fileLoc = getFileWritingLocation(yamlConfig, agentType=agentType)
        plots = fileLoc + "/plots/"
        os.makedirs(plots, exist_ok=True)
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Returns (%)")
        plt.title(f"Cumulative Returns Across Seeds – {plottedName}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plots}/Cumulative_Returns_Mean_{safe_filename}")
        plt.close()


########


def meanStatisticsTabulated(yamlConfig, bestTestSetPerformance, avRandReturn, rewards):
    # AI supported
    meanStdPerReward = {}

    for rewardFunc in rewards:
        allTrajectories = []
        mdds = []
        sharpes = []
        scores = []

        for seed in yamlConfig["varied_base_seeds"]:
            traj = np.array(bestTestSetPerformance[(rewardFunc, seed)][-1])

            normTraj = traj / yamlConfig["env"]["start_cash"] * 100 - 100
            allTrajectories.append(normTraj)

            raw = traj
            mdd = maxDrawdown(raw) * 100.0  # percent
            rets = np.diff(raw) / raw[:-1]
            sharpe = (np.mean(rets) / np.std(rets)) if np.std(rets) != 0 else 0.0

            mdds.append(mdd)
            sharpes.append(sharpe)

            metrs = scoreFormula(traj, avRandReturn, yamlConfig=yamlConfig)
            scores.append(metrs["Score"])

        allTrajectories = np.array(allTrajectories)
        mean_traj = np.mean(allTrajectories, axis=0)

        meanStdPerReward[rewardFunc] = {
            "final_mean_return": mean_traj[-1],  # %
            "final_std_dev": np.std(allTrajectories[:, -1]),  # % (across seeds)
            "mean_mdd": float(np.mean(mdds)),  # %
            "mean_sharpe": float(np.mean(sharpes)),
            "mean_score": float(np.mean(scores)),
        }
    rows = []
    for rewardFunc in rewards:
        properName = rewardFunc.replace("_", " = ") if "_" in rewardFunc else rewardFunc
        s = meanStdPerReward[rewardFunc]
        rows.append(
            {
                "Reward Function": properName,  # non-numeric → string/categorical
                "Final Mean Return (%)": s["final_mean_return"],
                "Std Dev (%)": s["final_std_dev"],
                "Mean MDD (%)": s["mean_mdd"],
                "Mean Sharpe": s["mean_sharpe"],
                "Mean Score": s["mean_score"],
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Reward Function",
            "Final Mean Return (%)",
            "Std Dev (%)",
            "Mean MDD (%)",
            "Mean Sharpe",
            "Mean Score",
        ],
    )

    df["Reward Function"] = df["Reward Function"].astype("string")
    num_cols = [c for c in df.columns if c != "Reward Function"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    df = df.sort_values("Mean Score", ascending=False, ignore_index=True)

    df_round = df.copy()
    df_round[num_cols] = df_round[num_cols].round(4)

    tablePath = getFileWritingLocation(yamlConfig) + "/tables/"
    os.makedirs(tablePath, exist_ok=True)

    with open(tablePath + "mean_stats.tex", "w") as f:
        f.write(df_round.to_latex())

    print(format_latex_table(df_round))


def finalIndexComparisonPlot(
    yamlConfig,
    bestTestSetPerformance,
    averPerformance,
    benchmarkPortVals,
    rewards,
    agentType="ppo",
    FIG_SIZE=(12, 8),
):
    plt.figure(figsize=FIG_SIZE)
    availableColors = ["purple", "darkgreen", "darkgray", "black", "navy"]
    colorCycle = cycle(availableColors)
    # Plot mean RL trajectories across seeds
    for rewardFunc in rewards:
        allTrajectories = []
        for seed in yamlConfig["varied_base_seeds"]:
            traj = (
                np.array(bestTestSetPerformance[(rewardFunc, seed)][-1])
                / yamlConfig["env"]["start_cash"]
                * 100
                - 100
            )
            allTrajectories.append(traj)
        meanTrajectory = np.mean(allTrajectories, axis=0)
        timesteps = np.arange(len(meanTrajectory))

        # Format name for legend
        properName = rewardFunc.split("_")
        if len(properName) > 1:
            if "CVaR" in properName[0]:
                plottedName = f"CVaR ($\\zeta={properName[1]}$)"
            elif "Differential Sharpe Ratio" in properName[0]:
                plottedName = f"Differential Sharpe Ratio ($\\eta={properName[1]}$)"
            else:
                plottedName = f"{properName[0]} ({properName[1]})"
        else:
            plottedName = properName[0]

        plt.plot(timesteps, meanTrajectory, label=f"{plottedName} (Mean)", linewidth=1)

    for key, value in benchmarkPortVals.items():
        traj = np.array(value) / yamlConfig["env"]["start_cash"] * 100 - 100
        color = next(colorCycle)
        plt.plot(traj, label=key, color=color, linewidth=1)
    plt.plot(
        np.array(averPerformance) / yamlConfig["env"]["start_cash"] * 100 - 100,
        label="Random (Baseline)",
        color="grey",
    )

    fileLoc = getFileWritingLocation(yamlConfig, agentType=agentType)
    plots = fileLoc + "/plots/"
    os.makedirs(plots, exist_ok=True)

    # Add baseline and labels
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns (%)")
    plt.title("Mean (Best) Agent Returns vs Index Strategies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plots}/Mean_Returns_vs_Indices")
    plt.close()
