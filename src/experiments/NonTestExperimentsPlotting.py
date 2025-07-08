import os
import numpy as np
import matplotlib.pyplot as plt
from MetricComputations import computeWeightedAUC


def plotNormalisationExpPerformance(
    folderPath,
    title="Mean Training Rewards & Validation Performance Over Training Period",
    saveFile=None,
):
    """
    Reads data from numbered subdirectories under folderPath, computes the mean and standard deviation
    across the runs for training rewards and validation performances, and plots the result.

    Parameters:
    - folderPath (str): The path to the folder containing numbered subdirectories for each run.
    - title (str): Plot title.
    - saveFile (str or None): If provided, the figure is saved to this file.

    Returns:
    - fig (matplotlib.figure.Figure): The created figure.
    """
    # Initialize containers.
    trainingRewardsAll = []
    validationPerformancesAll = []
    trainingRewardsStdAll = []
    validationPerformancesStdAll = []

    numDirectories = 0

    # Iterate through numeral directories in folderPath.
    for number in os.listdir(folderPath):
        numberDir = os.path.join(folderPath, number)
        if os.path.isdir(numberDir):
            trainingRewardsPath = os.path.join(numberDir, "trainingRewards.txt")
            validationPerformancesPath = os.path.join(
                numberDir, "validationPerformances.txt"
            )

            if os.path.exists(trainingRewardsPath) and os.path.exists(
                validationPerformancesPath
            ):
                trainingRewards = np.loadtxt(trainingRewardsPath)
                validationPerformances = np.loadtxt(validationPerformancesPath)

                # Initialize on the first valid directory.
                if numDirectories == 0:
                    trainingRewardsAll = np.zeros_like(trainingRewards)
                    validationPerformancesAll = np.zeros_like(validationPerformances)
                    trainingRewardsStdAll = []
                    validationPerformancesStdAll = []

                # Sum values element-wise.
                trainingRewardsAll += trainingRewards
                validationPerformancesAll += validationPerformances

                # Store arrays for std computation.
                trainingRewardsStdAll.append(trainingRewards)
                validationPerformancesStdAll.append(validationPerformances)

                numDirectories += 1
            else:
                print(
                    f"Skipping directory {numberDir} as it does not contain the required files."
                )

    # Compute means.
    trainingRewardsMean = trainingRewardsAll / numDirectories
    validationPerformancesMean = validationPerformancesAll / numDirectories

    # Compute standard deviations.
    trainingRewardsStd = np.std(trainingRewardsStdAll, axis=0)
    validationPerformancesStd = np.std(validationPerformancesStdAll, axis=0)

    # Rescale Training Rewards if lengths differ.
    validationLength = len(validationPerformancesMean)
    trainingLength = len(trainingRewardsMean)
    if trainingLength < validationLength:
        xOld = np.linspace(0, 1, trainingLength)
        xNew = np.linspace(0, 1, validationLength)
        trainingRewardsMean = np.interp(xNew, xOld, trainingRewardsMean)
        trainingRewardsStd = np.interp(xNew, xOld, trainingRewardsStd)

    # Plotting.
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = "tab:blue"
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Mean Training Rewards", color=color)
    line1 = ax1.plot(
        trainingRewardsMean, color=color, label="Mean Training Rewards", linewidth=2
    )
    ax1.fill_between(
        range(len(trainingRewardsMean)),
        trainingRewardsMean - trainingRewardsStd,
        trainingRewardsMean + trainingRewardsStd,
        color=color,
        alpha=0.2,
    )
    ax1.tick_params(axis="y", labelcolor=color)

    # Secondary axis for validation performance.
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Mean Validation Performance", color=color)
    line2 = ax2.plot(
        validationPerformancesMean,
        color=color,
        label="Mean Validation Performance",
        linewidth=2,
        linestyle="dashed",
    )
    ax2.fill_between(
        range(len(validationPerformancesMean)),
        validationPerformancesMean - validationPerformancesStd,
        validationPerformancesMean + validationPerformancesStd,
        color=color,
        alpha=0.2,
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # Combine legends from both axes.
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=10)

    fig.tight_layout()
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    if saveFile:
        plt.savefig(saveFile)
    plt.show()
    return fig


def loadResults(baseDir, noiseLevel, variedBaseSeeds):
    """Load performance arrays for a noise level across all seeds."""
    allRuns = []
    for seed in variedBaseSeeds:
        filePath = f"{baseDir}{seed}/{noiseLevel}.txt"
        if os.path.exists(filePath):
            data = np.loadtxt(filePath)
            processed = (data - 1) * 100
            allRuns.append(processed)
    return allRuns


def smooth(data, window=1):
    """Apply simple moving average."""
    if len(data) >= window:
        return np.convolve(data, np.ones(window) / window, mode="valid")
    return data


meanPercentageAboves = []
aucResults = []

# Variables to track best normalized curve
bestAuc = -np.inf
bestNoiseLevel = None
bestCurveX = None
bestCurveY = None


def plotNoiseComparison(yamlConfig, noiseLevel, env):
    global bestAuc, bestNoiseLevel, bestCurveX, bestCurveY

    normFolder = "portfolios/noises/Normalisation/"
    nonNormFolder = "portfolios/noises/NonNormalisation/"

    normResults = loadResults(normFolder, noiseLevel, yamlConfig["varied_base_seeds"])
    nonNormResults = loadResults(
        nonNormFolder, noiseLevel, yamlConfig["varied_base_seeds"]
    )

    if not normResults or not nonNormResults:
        print(f"Skipping σ={noiseLevel} — missing data.")
        return

    # Compute stats
    normMean = np.mean(normResults, axis=0)
    normStd = np.std(normResults, axis=0)
    nonNormMean = np.mean(nonNormResults, axis=0)
    nonNormStd = np.std(nonNormResults, axis=0)

    # Smooth
    normMeanSmooth = smooth(normMean)
    normStdSmooth = smooth(normStd)
    nonNormMeanSmooth = smooth(nonNormMean)
    nonNormStdSmooth = smooth(nonNormStd)

    percentageAbove = np.mean(normMeanSmooth > nonNormMeanSmooth) * 100
    meanPercentageAboves.append(percentageAbove)

    # Time axis
    xValues = (
        np.linspace(
            1,
            yamlConfig["epochs"] * env.datasetsAndDetails["training_windows"],
            len(normMeanSmooth),
        )
        * env.datasetsAndDetails["episode_length"]
    )

    # Compute weighted AUC for normalized only
    aucNorm = computeWeightedAUC(normMeanSmooth, xValues, weightPower=1.0)
    print(f"σ={noiseLevel}: Weighted AUC (Normalized) = {aucNorm:.2f}")
    aucResults.append((noiseLevel, aucNorm))

    # Track best normalized curve
    if aucNorm > bestAuc:
        bestAuc = aucNorm
        bestNoiseLevel = noiseLevel
        bestCurveX = xValues
        bestCurveY = normMeanSmooth

    # Plot comparison curves
    plt.figure(figsize=(10, 6))
    plt.plot(xValues, normMeanSmooth, label="Normalized", linewidth=2, color="blue")
    plt.plot(
        xValues, nonNormMeanSmooth, label="Non-Normalized", linewidth=2, color="orange"
    )

    # if SHOW_STD:
    #     plt.fill_between(xValues, normMeanSmooth - normStdSmooth, normMeanSmooth + normStdSmooth, color='blue', alpha=0.3)
    #     plt.fill_between(xValues, nonNormMeanSmooth - nonNormStdSmooth, nonNormMeanSmooth + nonNormStdSmooth, color='orange', alpha=0.3)

    plt.title(f"Noise Level σ={noiseLevel} Comparison")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Cumulative Return (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/noiseComparison_sigma_{noiseLevel}.png")
    plt.close()
    print(f"Saved plot for σ={noiseLevel}.")


def runNoiseComparison(yamlConfig, env):
    for noise in yamlConfig["noises"]:
        plotNoiseComparison(yamlConfig, noise, env)

    print(
        f"Generally, the normalisation makes performance over the training period: {round(np.mean(meanPercentageAboves), 2)} %) of the time better than the non-normalised data."
    )
    print("Should normalise = ", np.mean(meanPercentageAboves) > 50)

    # After plotting all noise levels
    aucResults.sort(key=lambda x: x[1], reverse=True)
    print("\n--- Weighted AUC (Normalized) Summary ---")
    for sigma, auc in aucResults:
        print(f"σ={sigma:.3f} -> AUC: {auc:.2f}")

    # Plot the best normalized curve by weighted AUC
    if bestCurveX is not None and bestCurveY is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(
            bestCurveX,
            bestCurveY,
            label=f"Best Normalized Noise σ={bestNoiseLevel}",
            linewidth=3,
            color="green",
        )
        plt.xlabel("Training Timesteps")
        plt.ylabel("Average Cumulative Return (%)")
        plt.title("Best Normalized Learning Curve by Weighted AUC")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
