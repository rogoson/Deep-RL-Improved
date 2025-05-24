import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces

device = torch.device("cpu")
LOGGING = False  # Set to True to enable logging for debugging purposes


class TimeSeriesEnvironment(gym.Env):
    def __init__(
        self,
        marketData,
        normData,
        TIME_WINDOW,
        EPISODE_LENGTH,
        startCash,
        AGENT_RISK_AVERSION,
        transactionCost=0.001,
    ):

        self.marketData = marketData  # Stores percentage price changes over time
        self.normData = normData  # normalized data for the agent to use
        self.normDataTensors = {  # Tensorized normalized data for the agent to use
            k: torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32, device=device)
            for k, df in self.normData.items()  # The -1 ignores the return collumn, whicih is the percentage change
            # between the current and previous time step.
        }

        self.TIME_WINDOW = TIME_WINDOW
        self.timeStep = 0
        self.episodeLength = EPISODE_LENGTH
        self.transactionCost = transactionCost
        self.allocations = []  # For transaction Cost Calculation
        self.startCash = startCash
        self.previousPortfolioValue = startCash
        self.RETURNS = [0]
        self.PORTFOLIO_VALUES = [self.startCash]
        self.AGENT_RISK_AVERSION = AGENT_RISK_AVERSION
        self.isReady = False

        self.CVaR = [0]
        self.maxAllocationChange = 1  # liquidigy parameter.

        # Required for Differential Sharpe Ratio
        self.decayRate = 0.01
        self.meanReturn = None
        self.meanSquaredReturn = None

    def getData(self, timeStep=None):
        """
        Retrieves Data for the agent using the current timestep and the time window.
        :param timeStep: The current timestep of the environment - > TIME_WINDOW in every case to prevent negative indexing.
        :return: The data for the agent to use.
        """
        if timeStep is None:
            timeStep = self.timeStep
        data = self.getMarketData(
            timeStep,
            self.TIME_WINDOW,
        )
        return data

    def getMarketData(self, i, TIME_WINDOW):
        """
        Creates the observation matrix for the agent to use. This is a matrix of size (TIME_WINDOW, numFeatures).
        :param i: The current timestep of the environment.
        :param TIME_WINDOW: The time window of the environment.
        :return: The observation matrix for the agent to use.
        The observation matrix is sort of like a rolled up summary of what has happened in the last TIME_WINDOW timesteps.
        """
        # SHOULD NOT be trying to get market data if this is the case - will leak future info
        # unforgivable stupidity #2 - fixed. Indexing errors, along with correct LSTM direction.
        if i < TIME_WINDOW:
            return
        observationMatrix = []
        if LOGGING:
            print(f"\n=== getMarketData at timestep i = {i} ===")
        for p in range(1, TIME_WINDOW + 1):
            index = i - p + 1
            if LOGGING:
                print(f"[p = {p}] → index = {index}")

            allocations = self.allocations[index]
            if LOGGING:
                print(f"  Allocations[{index}] = {allocations}")
            portfolioValue = torch.tensor(
                [self.PORTFOLIO_VALUES[index]], dtype=torch.float32, device=device
            )
            if LOGGING:
                print(f"  PortfolioValue[{index}] = {portfolioValue}")

            obsVector = [allocations, portfolioValue]

            for name, dfTensor in self.normDataTensors.items():
                dataRow = dfTensor[
                    index
                ]  # starts from current time step and goes back TIME_WINDOW steps
                if LOGGING:
                    print(f"  MarketData[{name}][{index}] = {dataRow.cpu().numpy()}")
                obsVector.append(dataRow)

            combined = torch.cat(obsVector).to(device)
            observationMatrix.insert(
                0, combined
            )  # to ensure the correct order is passed to LSTM. Oldest will be at the top. No idea how I messed this up before.

        return torch.stack(observationMatrix).unsqueeze(0)

    def step(
        self, action, rewardMethod="CVaR", returnNextObs=True
    ):  # if random, no need to return next obs
        newPortfolioValue = self.calculatePortfolioValue(
            action,
            self.marketData.iloc[self.timeStep + 1].values,
        )

        reward = newPortfolioValue - self.PORTFOLIO_VALUES[-1]
        self.timeStep += 1
        info = dict()

        done = False

        # below not really needed if using indexes [virtually impossible to lose all your money]
        if newPortfolioValue / self.startCash < 0.7:
            done = True
            info["reason"] = "portfolio_below_70%"
        elif self.timeStep + 1 == self.episodeLength:
            done = True
            info["reason"] = "max_steps_reached"

        self.RETURNS.append(reward)
        self.PORTFOLIO_VALUES.append(newPortfolioValue)
        if rewardMethod == "CVaR":
            reward = self.getCVaRReward(reward)
        elif rewardMethod == "Standard Logarithmic Returns":
            reward = self.getCVaRReward(reward, False)
        elif "Differential" in rewardMethod:
            reward = self.calculateDifferentialSharpeRatio(reward)
        else:
            raise ValueError("Unknown reward method: " + rewardMethod)
        # unforgivable stupidity #1 - fixed - I removed this.
        # if info.get("reason") == "portfolio_below_70%":
        #     reward -= 100 * abs(reward)  # big penalty for loss of 30%

        if not done and returnNextObs:
            nextObs = self.getData(self.timeStep)
        else:
            nextObs = None
        return (
            nextObs,
            reward,
            done,
            False,
            info,
        )

    def getMetrics(self, portfolioValues=None, returns=None):
        """
        Some metrics that can be returned for a given run.
        """
        if portfolioValues == None:
            portfolioValues = self.PORTFOLIO_VALUES
        info = dict()
        info["Cumulative \nReturn (%)"] = round(
            100 * (portfolioValues[-1] / self.startCash) - 100, 2
        )
        info["Maximum \nDrawdown (%)"] = self.maxDrawdown()
        percChange = np.diff(portfolioValues) / portfolioValues[:-1]
        info["Sharpe Ratio"] = (
            round(np.mean(percChange) / np.std(percChange), 4)
            if np.std(percChange) != 0
            else 0.0
        )
        info["Total Timesteps"] = self.timeStep
        return info

    def maxDrawdown(self, portfolioValues=None):
        """
        maximum drawdown calculation
        :param portfolioValues: The portfolio values to calculate the maximum drawdown for.
        :return: The maximum drawdown for the portfolio values.
        """
        if portfolioValues == None:
            portfolioValues = self.PORTFOLIO_VALUES
        maxValue = float("-inf")
        maxDrawdown = 0.0
        for value in portfolioValues:
            maxValue = max(maxValue, value)
            drawdown = (maxValue - value) / maxValue * 100
            maxDrawdown = max(maxDrawdown, drawdown)
        return maxDrawdown

    def calculateDifferentialSharpeRatio(self, currentReturn):
        """
        In line with Moody & Saffel's "Reinforcement Learning for Trading" 1998 Paper
        It is best to look at the formula found in this paper:
            https://papers.nips.cc/paper_files/paper/1998/file/4e6cd95227cb0c280e99a195be5f6615-Paper.pdf
        The relevant equations are found on page 3.
        """
        if self.meanReturn is None:
            self.meanReturn = currentReturn
            self.meanSquaredReturn = currentReturn**2
            return 0.0

        prevMeanReturn = self.meanReturn
        prevMeanSquaredReturn = self.meanSquaredReturn

        deltaMean = currentReturn - prevMeanReturn
        deltaSquared = currentReturn**2 - prevMeanSquaredReturn
        self.meanReturn += self.decayRate * deltaMean
        self.meanSquaredReturn += self.decayRate * deltaSquared

        denom = (prevMeanSquaredReturn - prevMeanReturn**2) ** 1.5
        if denom == 0:
            return 0.0
        numerator = (
            prevMeanSquaredReturn * deltaMean - 0.5 * prevMeanReturn * deltaSquared
        )
        differentialSharpeRatio = numerator / denom
        return differentialSharpeRatio

    def normaliseValue(self, value):
        return np.sign(value) * (np.log1p(np.abs(value)))

    def getCVaRReward(self, r, useCVaR=True):
        """
        CVaR reward calculation.
        :param r: The reward to be calculated.
        :param useCVaR: Whether to use CVaR or not. If not, the reward is just the normalised value of the reward.
        """
        if useCVaR and self.AGENT_RISK_AVERSION != 0:
            currentCVaR = self.calculateCVaR()
            changeInCVaR = currentCVaR - self.CVaR[-1]
            cVaRNum = self.normaliseValue(changeInCVaR)
            riskPenalty = self.AGENT_RISK_AVERSION * cVaRNum
            self.CVaR.append(currentCVaR)
        else:
            riskPenalty = 0
        scaledReward = self.normaliseValue(r)
        finalReward = scaledReward - riskPenalty
        return finalReward

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)
        self.timeStep = 0
        self.allocations = []
        self.previousPortfolioValue = None
        self.PORTFOLIO_VALUES = [self.startCash]
        self.isReady = False
        self.CVaR = [0]
        self.RETURNS = [0]
        self.meanReturn = None
        self.meanSquaredReturn = None

    def calculatePortfolioValue(self, targetAllocation, closingPriceChanges):
        """
        Portfoloio Value Calculation Logic. This is detailed in section 3.1.1 of the paper.
        """
        if not isinstance(targetAllocation, torch.Tensor):
            targetAllocation = torch.tensor(
                targetAllocation, dtype=torch.float32, device=device
            )

        if self.previousPortfolioValue is None:
            self.previousPortfolioValue = self.startCash

        if self.allocations:
            prevAllocation = self.allocations[-1]
            currentAllocation = (
                1 - self.maxAllocationChange
            ) * prevAllocation + self.maxAllocationChange * targetAllocation
        else:
            prevAllocation = torch.zeros(len(closingPriceChanges) + 1, device=device)
            prevAllocation[0] = 1  # all cash
            currentAllocation = targetAllocation
            self.allocations.append(
                prevAllocation
            )  # for things to line up - only occurs once

        if not isinstance(closingPriceChanges, torch.Tensor):
            closingPriceChanges = torch.tensor(
                closingPriceChanges, dtype=torch.float32, device=device
            )
        # 0 for cash, presumed not to change
        closingPriceChanges = torch.cat(
            [torch.tensor([0.0], device=device), closingPriceChanges]
        )

        wealthDistribution = self.previousPortfolioValue * currentAllocation
        changeWealth = (1 + closingPriceChanges) * wealthDistribution

        transactionCost = 0
        if self.transactionCost > 0:
            transactionCost = self.transactionCost * torch.sum(
                self.previousPortfolioValue
                * torch.abs(currentAllocation - prevAllocation)
            )

        portfolioValue = torch.sum(changeWealth) - transactionCost
        self.previousPortfolioValue = portfolioValue.item()
        self.allocations.append(currentAllocation)

        return self.previousPortfolioValue

    def calculateCVaR(self, percentage=0.05):
        """
        Actual CVaR calculation. This is done by sorting the returns and taking the mean of the worst 'percentage' of returns.
        CVaR returns 0 for the max(10, 1/percentage) timesteps, as there is not enough data to reasonably calculate it.
        This is to prevent early instability in the training process.
        """
        if len(self.CVaR) < max(10, int(1 / percentage)):
            return np.mean(self.CVaR)
        sortedReturns = sorted(self.RETURNS)
        indexToBePicked = int(np.ceil(percentage * len(sortedReturns)))
        CVaR = np.mean(sortedReturns[:indexToBePicked])
        return CVaR

    def render(self, mode="human"):
        # not implemented
        pass

    def close(self):
        pass

    def getIsReady(self):
        """
        These let the training loop know when the environment is ready to be used.
        """
        return self.isReady

    def setIsReady(self, boolean: bool):
        self.isReady = boolean
