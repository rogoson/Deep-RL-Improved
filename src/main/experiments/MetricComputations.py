import numpy as np


def computeWeightedAUC(yValues, xValues=None, weightPower=1.0):
    """
    Computes a weighted Area Under Curve (AUC) for a learning curve. Helps to score noise levels
    """
    yValues = np.asarray(yValues)
    if xValues is None:
        xValues = np.arange(len(yValues))
    xValues = np.asarray(xValues)
    normTime = (xValues - xValues.min()) / (xValues.max() - xValues.min())
    weights = normTime**weightPower
    weightedY = yValues * weights
    auc = np.trapezoid(weightedY, xValues)
    totalWeight = np.trapezoid(weights, xValues)
    normalizedAUC = auc / totalWeight if totalWeight > 0 else 0
    return normalizedAUC


def maxDrawdown(arr):
    # Maximum Drawdown calculation
    maxValue = float("-inf")
    maxDrawdown = 0.0
    for value in arr:
        maxValue = max(maxValue, value)
        drawdown = (maxValue - value) / maxValue
        maxDrawdown = max(maxDrawdown, drawdown)
    return maxDrawdown


# Scoring Formula
def scoreFormula(agentArray, averageRandomReturn, yamlConfig):
    # score by (cumulative return - average random return)/max drawdown all times sharpe ratio
    cumulativeReturn = agentArray[-1] / yamlConfig["env"]["start_cash"] - 1
    maximumDrawdown = maxDrawdown(agentArray)
    percChange = np.diff(agentArray) / agentArray[:-1]
    sharpe = np.mean(percChange) / np.std(percChange) if np.std(percChange) != 0 else 0
    score = ((cumulativeReturn - averageRandomReturn) / maximumDrawdown) * np.abs(
        sharpe
    )
    metrics = {
        "Cumulative \nReturn (%)": cumulativeReturn * 100,
        "Maximum \nDrawdown (%)": maximumDrawdown * 100,
        "Sharpe Ratio": sharpe,
        "Score": score,
    }
    return metrics
