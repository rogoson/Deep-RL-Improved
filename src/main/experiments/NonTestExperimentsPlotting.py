import os
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from .MetricComputations import computeWeightedAUC
from main.utils.GeneralUtils import getFileWritingLocation


def loadResults(baseDir, variedBaseSeeds):
    """Load performance arrays for a param across all seeds."""
    allRuns = []
    for seed in variedBaseSeeds:
        filePath = f"{baseDir}/{seed}/validationPerformances.txt"
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


def plotParameterComparison(yamlConfig, parameterDict, env, agentType="ppo"):
    aucResults = []

    # Variables to track best normalized curve
    bestAuc = -np.inf
    bestParameter = None
    bestCurveX = None
    bestCurveY = None

    for value in next(iter(parameterDict.values())):
        parameter = next(iter(parameterDict.keys()))
        baseFolder = getFileWritingLocation(yamlConfig, agentType=agentType)
        normFolder = f"{baseFolder}/portfolios/{parameter}/{value}"

        normResults = loadResults(normFolder, yamlConfig["varied_base_seeds"])
        if len(normResults) == 0:
            print(
                f"Data not found/Unreadable for this experiment. Attempted path was: {normFolder}"
            )
            return

        # Compute stats
        normMean = np.mean(normResults, axis=0)
        normMeanSmooth = smooth(normMean)

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
        auc = computeWeightedAUC(normMeanSmooth, xValues)
        print(f"Value={parameter}: Weighted AUC ({value}) = {auc:.2f}")
        aucResults.append((value, auc))

        # Track best normalized curve
        if auc > bestAuc:
            bestAuc = auc
            bestParameter = value
            bestCurveX = xValues
            bestCurveY = normMeanSmooth

        os.makedirs(f"{baseFolder}/plots/", exist_ok=True)
        # Plot comparison curves
        plt.figure(figsize=(10, 6))
        plt.plot(
            xValues,
            normMeanSmooth,
            label="Validation Returns",
            linewidth=2,
            color="blue",
        )

        plt.title(f"Param Value={parameter.title()} | {value} Comparison")
        plt.xlabel("Training Timesteps")
        plt.ylabel("Average Cumulative Return (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"{baseFolder}/plots/{"_".join(parameter.split())}Comparison_{value}.png",
        )
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        print(f"Saved plot for param={parameter}.")
    return {
        "aucResults": aucResults,
        "bestCurveX": bestCurveX,
        "bestCurveY": bestCurveY,
        "bestParameter": bestParameter,
    }


def displayResults(aucResults, bestCurveX, bestCurveY, bestParameter):
    # After plotting all noise levels
    aucResults.sort(key=lambda x: x[1], reverse=True)
    print("\n--- Weighted AUC Summary ---")
    for param, auc in aucResults:
        print(f"Value={param} -> AUC: {auc:.2f}")

    # Plot the best normalized curve by weighted AUC
    if bestCurveX is not None and bestCurveY is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(
            bestCurveX,
            bestCurveY,
            label=f"Best Parameter={bestParameter}",
            linewidth=3,
            color="green",
        )
        plt.xlabel("Training Timesteps")
        plt.ylabel("Average Cumulative Return (%)")
        plt.title("Best Normalized Learning Curve by Weighted AUC")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def runParameterComparison(yamlConfig, env, agentType="ppo"):

    configDir = (
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "configs"
    )

    parameterValueMappings = {
        "FEATURE OUTPUT SIZE": yamlConfig["hyperparameters"]["feature_output_sizes"],
        # "NORMALIZE DATA": ["True", "False"], # for another day :)
    }

    for usingLSTMFeature in [True, False]:
        print("=" * 50)
        print(f"Feature Extractor = {"LSTM" if usingLSTMFeature else "CNN"}")
        yamlConfig["usingLSTMFeatureExtractor"] = usingLSTMFeature
        print("=" * 50)
        for key, value in parameterValueMappings.items():
            resultsDict = plotParameterComparison(
                yamlConfig,
                {key: value},
                env,
                agentType=agentType,
            )

            displayResults(
                resultsDict["aucResults"],
                resultsDict["bestCurveX"],
                resultsDict["bestCurveY"],
                resultsDict["bestParameter"],
            )
            print("=" * 30)

            configYaml = configDir / "config.yaml"
            with configYaml.open("r") as f:
                data = yaml.safe_load(f)

            if key == "FEATURE OUTPUT SIZE":
                agentDict = data.setdefault("agent", {})
                typeDict = agentDict.setdefault(agentType, {})
                featureDict = typeDict.setdefault("best_feature_output_sizes", {})
                featureKey = "cnn" if not usingLSTMFeature else "lstm"
                featureDict[featureKey] = resultsDict["bestParameter"]
            # else:
            #     boolMap = {"True": True, "False": False}
            #     data["normalise_data"] = boolMap[resultsDict["bestParameter"]]

            with configYaml.open("w") as f:
                yaml.dump(data, f)
