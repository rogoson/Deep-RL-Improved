import numpy as np
import gymnasium as gym
import torch
import os
import warnings
from main.utils.GeneralUtils import normData
from pathlib import Path
import pandas as pd
import yfinance as yf
from functools import reduce
import talib as ta
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
# Convert all UserWarnings into exceptions
BASE_DIR = Path(__file__).parent
warnings.filterwarnings("error", category=RuntimeWarning)

LOGGING_MARKET_DATA = (
    False  # Set to True to enable logging_MARKET_DATA for debugging purposes
)
LOGGING_CVAR_REWARD = False
LOGGING_LOG_REWARD = False
LOGGING_DSR_REWARD = False


class TimeSeriesEnvironment(gym.Env):
    def __init__(
        self,
        TIME_WINDOW,
        startCash,
        normaliseData,
        perturbationNoise,
        agentRiskAversion=0,
        transactionCost=0.001,
        render_config=None,
    ):

        self.marketData = None  # Stores percentage price changes over time
        self.normDataTensors = None  # Normalised data tensors for the agent

        self.TIME_WINDOW = TIME_WINDOW
        self.timeStep = 0
        self.episodeLength = None
        self.transactionCost = transactionCost
        self.allocations = []  # For transaction Cost Calculation
        self.startCash = startCash
        self.previousPortfolioValue = startCash
        self.PORTFOLIO_DIFFERENCES = [0]
        self.PORTFOLIO_VALUES = [self.startCash]
        self.agentRiskAversion = agentRiskAversion
        self.isReady = False
        self.hasVisualised = False
        self.baseSeed = None
        self.perturbationNoise = perturbationNoise
        self.normaliseData = normaliseData

        self.animationNumber = None

        self.CVaR = [0]
        self.maxAllocationChange = 1  # liquidigy parameter.

        # Required for Differential Sharpe Ratio
        self.decayRate = None
        self.meanReturn = None
        self.meanSquaredReturn = None

        self.datasetsAndDetails = None
        self.productIds = None
        self.index = None

    def setup(self, yamlConfig):
        data, returnableInfo = self.createDataframes(yamlConfig)
        self.partitionData(data, yamlConfig)
        return returnableInfo

    def createDataframes(self, configuration):
        dataframes = dict()

        period = configuration["env"]["period"]

        index = configuration["active_index"]
        self.index = index
        productIds = configuration["env"]["tickers"][index]
        self.productIds = productIds
        if len(productIds) == 0:
            raise ValueError("No product IDs provided for data retrieval.")

        self.numberOfAssets = len(productIds)
        redownloadData = configuration["env"]["redownload"]
        self.baseSeed = configuration["env"]["base_seed"]

        """
        NOTE ON DATA CONSISTENCY:
        Yahoo finance value precision still a problem, but can be partly mitigated by rounding prices to 2dp.
        I tried rounding indicators to 4 significant figures. 4 is arbitrary, but higher values
        will likely lead to more instability.
        Later testing revealed that these mitigation techniques helped a bit, but were not perfect. I did my best.

        """
        #########################################################################################

        def retrieveIndexData(ticker, verbose=True):
            # Define date range
            startDate = configuration["env"]["start_date"]
            endDate = datetime.today().strftime("%Y-%m-%d")

            ohlcData = {}
            try:
                stockData = yf.download(ticker, start=startDate, end=endDate)

                # To have a more reliable indicator implementation. Mine were fine I believe, but the below is safer.
                ohlcData[ticker] = stockData[
                    ["Low", "High", "Open", "Close", "Volume"]
                ].copy()

                # Also round to 2dp to help reduce instability
                ohlcData[ticker][["Low", "High", "Open", "Close"]] = ohlcData[ticker][
                    ["Low", "High", "Open", "Close"]
                ].round(2)

                high = np.round(
                    stockData["High"].values.astype(np.float64).flatten(), 2
                )
                low = np.round(stockData["Low"].values.astype(np.float64).flatten(), 2)
                close = np.round(
                    stockData["Close"].values.astype(np.float64).flatten(), 2
                )

                # arbitrary, but popular indicators -> Subset of those used between Soleymani & Zou.
                ohlcData[ticker]["ATR"] = ta.ATR(high, low, close, timeperiod=14)
                ohlcData[ticker]["Momentum"] = ta.MOM(
                    close, timeperiod=1
                )  # Momentum is used by Soleymani, not (RoC=Momentum Oscillator, my mistake)
                ohlcData[ticker]["CCI"] = ta.CCI(high, low, close, timeperiod=20)
                macd, signal, hist = ta.MACD(
                    close, fastperiod=12, slowperiod=26, signalperiod=9
                )
                ohlcData[ticker][
                    "MACD"
                ] = macd  # histogram isn't the same as the MACD line, also used by Zou, not Soleymani
                ohlcData[ticker]["EMA"] = np.round(
                    ta.EMA(close, timeperiod=30), 2
                )  # Exponential Moving Average is a price moving average, so it should be rounded to 2dp as well

                if verbose:
                    print(f"\n{'-'*40}")
                    print(
                        f"Attempted to retrieve data for {ticker} from {startDate} to {endDate}"
                    )
                    print(
                        f"Actually retrieved data for {ticker} from {stockData.index[0].strftime('%Y-%m-%d')} to {stockData.index[-1].strftime('%Y-%m-%d')}"
                    )
                    print(f"Data shape: {ohlcData[ticker].shape}")
                    print(f"\n{ticker} - Indicator Data Lengths:")
                    print(f"{'-'*40}")
                    print(f"{'Close:':<20} {len(close)}")
                    print(f"{'ATR:':<20} {len(ohlcData[ticker]['ATR'])}")
                    print(f"{'CCI:':<20} {len(ohlcData[ticker]['CCI'])}")
                    print(f"{'Momentum:':<20} {len(ohlcData[ticker]['Momentum'])}")
                    print(f"{'EMA:':<20} {len(ohlcData[ticker]['EMA'])}")
                    print(f"{'MACD:':<20} {len(macd)}")
                    print(f"{'-'*40}\n")

            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")

            dfOhlc = pd.concat(ohlcData, axis=1)
            return dfOhlc

        def roundToSf(array, sigFigs=4):
            """
            Round a NumPy array or Pandas Series to the specified number of significant figures
            Decision of 4 sigfigs is arbitrary, but higher values will likely lead to instability.
            """
            return np.array(
                [
                    (
                        round(x, sigFigs - int(np.floor(np.log10(abs(x)))) - 1)
                        if x != 0
                        else 0
                    )
                    for x in array
                ]
            )

        def postProcessIndicators(df, columns):
            """
            Post-processes indicators - turns them to 4sf to promote stability
            """
            dfRounded = df.copy()
            for col in columns:
                if col in dfRounded.columns:
                    dfRounded[col] = roundToSf(dfRounded[col])
                else:
                    print(f"Warning: Column '{col}' not found in DataFrame.")
            return dfRounded

        today = datetime.today().strftime("%Y-%m-%d")
        # Create and write to a file
        if os.path.exists(BASE_DIR / "lastDownloaded.txt"):
            with open(BASE_DIR / "lastDownloaded.txt", "r+") as file:
                date = file.readline().strip()
                if date == today:
                    redownloadData = False
                else:
                    file.seek(0)
                    file.write(f"{today}\n")
                    file.truncate()

        else:
            with open(BASE_DIR / "lastDownloaded.txt", "w") as file:
                file.write(f"{today}\n")

        if redownloadData:
            for productId in productIds:
                """
                I was previously working with Crypto data also, so I implemented this to ensure
                column names are consistent across all dataframes.
                """
                dataframe = None

                try:
                    dataframe = retrieveIndexData(ticker=productId)
                except Exception:
                    continue

                if dataframe.empty:
                    redownloadData = False
                    if not os.path.exists(BASE_DIR / "CSVs/"):
                        raise Exception("Market Data not downloadable and not saved.")
                    break
                columnNames = [value[1] for value in list(dataframe.columns.values)]
                dataframe.columns = columnNames
                dataframes[productId] = dataframe

        def verifyDataConsistency(dataframes, stage="None"):
            """
            Strictly verifies that all dataframes have the same columns (in the same order)
            and the same index (ordered identically). Raises a ValueError if any inconsistency is found.

            Parameters:
                dataframes (dict): Dictionary where keys are identifiers (e.g., tickers) and values are pandas DataFrames.
                stage (str): A label (optional) to indicate at what stage this check is performed.

            Raises:
                ValueError: If any dataframe does not have the same columns or index as the first dataframe.
            """
            if not dataframes:
                raise ValueError("No dataframes to verify.")

            # Retrieve the columns and index from the first dataframe (in order)
            first_key = next(iter(dataframes))
            first_columns = list(dataframes[first_key].columns)
            first_index = list(dataframes[first_key].index)

            # Check each dataframe for consistency in columns and index
            for ticker, df in dataframes.items():
                # Ensure columns are identical in both names and order
                if list(df.columns) != first_columns:
                    raise ValueError(
                        f"Inconsistency at stage {stage}: DataFrame for '{ticker}' has columns {list(df.columns)} ; "
                        f"expected {first_columns}."
                    )
                # Ensure the index is identical in both values and order
                if list(df.index) != first_index:
                    raise ValueError(
                        f"Inconsistency at stage {stage}: DataFrame for '{ticker}' has index {list(df.index)} ; "
                        f"expected {first_index}."
                    )

            # If everything is consistent, you can optionally log success.
            print(
                f"All dataframes are consistent in terms of columns and index at stage {stage}."
            )

        def dropNaN(data, logDroppedRows=True):
            # Create new dictionaries to store the cleaned data and the dropped rows
            cleanedData = {}
            droppedRows = {}

            for ticker, df in data.items():
                # Capture the indices of the rows which will be dropped
                clean_df = df.dropna()
                dropped_idx = df.index.difference(clean_df.index)

                # Store the cleaned dataframe and the dropped rows (as a DataFrame)
                cleanedData[ticker] = clean_df
                droppedRows[ticker] = df.loc[dropped_idx]

                # Optionally, print the dropped rows for debugging/inspection.
                if logDroppedRows and not droppedRows[ticker].empty:
                    print(f"Dropped rows for ticker {ticker}:")
                    print(
                        droppedRows[ticker].head()
                    )  # Display only the first few rows for brevity
                    print("-" * 80)
            return cleanedData

        # Required for index data that pull stock data from different exchanges
        def commonaliseAndPreprocess(data):
            """
            Returns data, ensures that all dataframes have the same index (dates).
            Also post-processes indicators returns 4 sigfigs.
            """
            data = dropNaN(
                data
            )  # order of nan and commonalise switched - dropping NaNs first ensures that there are no gaps at beginning or end of data
            commonDates = reduce(
                lambda x, y: x.intersection(y), [df.index for df in data.values()]
            )
            for ticker, df in data.items():
                data[ticker] = df.reindex(commonDates)
                data[ticker]["Times"] = commonDates
            verifyDataConsistency(data, stage="Commonalise (Post NaN Drop)")
            data = {
                ticker: postProcessIndicators(
                    df, columns=["ATR", "Momentum", "CCI", "MACD"]
                )
                for ticker, df in data.items()
            }  # EMA is already rounded to 2dp, so we skip it here (as it is a price moving average)
            verifyDataConsistency(data, stage="Post-process Indicators")
            return data

        if redownloadData:
            dataframes = commonaliseAndPreprocess(dataframes)
            for product in productIds:
                """
                Generating dataframe for each product. Stored as marketdata.
                Now using reliable implementations of indicators from TA-Lib, as opposed to doing it myself.
                No longer have unstable indicators at the start of the dataframe, so we have 20 extra rows in total.
                """
                df = dataframes[product]
                df = df.drop(columns=["Open", "High", "Volume", "Low"])
                df = df.reset_index(drop=True)
                dataframes[product] = df
                if not os.path.exists(BASE_DIR / "CSVs/"):
                    os.makedirs(BASE_DIR / "CSVs/")
                df.to_csv(
                    BASE_DIR / f"CSVs/{product}_{period}_periods.csv",
                    sep="\t",
                    index=False,
                )
        else:
            for productId in productIds:
                dataframes[productId] = pd.read_csv(
                    BASE_DIR / f"CSVs/{productId}_{period}_periods.csv", sep="\t"
                )

        for product in productIds:
            """
            Dropping the times column - is not required for training.
            """
            df = dataframes[product]
            dataframes[product] = df.drop("Times", axis=1)

        self.numberOfFeatures = 2 + (
            1 + len((list(dataframes.values())[0]).columns)
        ) * len(self.productIds)

        detailsToReturn = {
            "numberOfAssets": self.numberOfAssets,
            "numberOfFeatures": self.numberOfFeatures,
        }

        return dataframes, detailsToReturn

    def partitionData(self, data, configuration):
        """
        Splits data into train, val, test portions. These are non-overlapping.
        Produce details and datasets
        """
        datasetSize = list(data.values())[0].shape[0]
        TRAINING_DATA = {}
        VALIDATION_DATA = {}
        TESTING_DATA = {}

        TRAINING_PERIODS = round(datasetSize * 2 / 3)
        VALIDATION_PERIODS = (datasetSize - TRAINING_PERIODS) // 2

        for key, df in data.items():
            trainSlice = df.iloc[:TRAINING_PERIODS].copy()
            validationSlice = df.iloc[
                TRAINING_PERIODS : TRAINING_PERIODS + VALIDATION_PERIODS
            ].copy()
            testSlice = df.iloc[TRAINING_PERIODS + VALIDATION_PERIODS :].copy()

            TRAINING_DATA[key] = trainSlice
            VALIDATION_DATA[key] = validationSlice
            TESTING_DATA[key] = testSlice

        TRAINING_VALIDATION_DATA = {}
        # Join Training + Validation data set
        for key, df in TRAINING_DATA.items():
            TRAINING_VALIDATION_DATA[key] = pd.concat(
                [TRAINING_DATA[key], VALIDATION_DATA[key].copy()]
            )
            TRAINING_VALIDATION_DATA[key] = TRAINING_VALIDATION_DATA[key].reset_index(
                drop=True
            )

        EPISODE_LENGTH = datasetSize // 3
        TIMESTEP_SHIFT = EPISODE_LENGTH // 10
        TRAINING_WINDOWS = ((TRAINING_PERIODS - EPISODE_LENGTH) // TIMESTEP_SHIFT) + 1
        TEST_TRAINING_WINDOWS = (
            (TRAINING_PERIODS + VALIDATION_PERIODS - EPISODE_LENGTH) // TIMESTEP_SHIFT
        ) + 1
        SUM_TRAINING_PERIODS = (
            TRAINING_WINDOWS
            * configuration["epochs"]
            * (EPISODE_LENGTH - configuration["time_window"])
        )  # because first time window steps are not used to learn
        SUM_TEST_TRAINING_PERIODS = (
            TEST_TRAINING_WINDOWS
            * configuration["epochs"]
            * (EPISODE_LENGTH - configuration["time_window"])
        )

        datasets = {
            "training_data": TRAINING_DATA,
            "validation_data": VALIDATION_DATA,
            "testing_data": TESTING_DATA,
            "training_validation_data": TRAINING_VALIDATION_DATA,
        }

        details = {
            "training_windows": TRAINING_WINDOWS,
            "test_training_windows": TEST_TRAINING_WINDOWS,
            "timestep_shift": TIMESTEP_SHIFT,
            "episode_length": EPISODE_LENGTH,
            "sum_training_periods": SUM_TRAINING_PERIODS,
            "sum_test_training_periods": SUM_TEST_TRAINING_PERIODS,
        }

        if not self.hasVisualised:
            self.visualiseData(datasets)

        datasets.update(details)
        self.datasetsAndDetails = datasets

        if not self.hasVisualised:
            self.demonstrateNoiseEffect(configuration["noises"])
        self.hasVisualised = True  # Ensure visualisation is only done once

    def visualiseData(self, datasets):
        # For each dataset type, plot normalized 'Close' prices from all keys on a single graph.
        for setName, dfDict in datasets.items():
            plt.figure(figsize=(12, 8))

            keys = list(dfDict.keys())

            # Set up a colormap with one color per key
            colourMap = plt.colormaps.get_cmap("nipy_spectral")
            colors = [colourMap(i / (len(keys) - 1)) for i in range(len(keys))]

            # Store normalized close values for averaging
            normalized_values = []

            # Plot individual key lines
            for idx, key in enumerate(keys):
                df = dfDict[key]
                normalizedClose = (
                    df["Close"] / df["Close"].iloc[0]
                )  # Normalize by first value
                plt.plot(
                    normalizedClose,
                    label=key,
                    alpha=0.7,
                    linestyle="solid",
                    linewidth=1,
                    color=colors[idx],
                )
                normalized_values.append(normalizedClose)

            # Compute and plot the average price movement
            avg_price_movement = sum(normalized_values) / len(normalized_values)
            plt.plot(
                avg_price_movement,
                label="Average Price Movement",
                color="black",
                linewidth=3,
                linestyle="dashed",
            )

            plt.xlabel("Time")
            plt.ylabel("Normalized Close Price")
            plt.title(
                f"{" ".join([part.capitalize() for part in setName.split("_")])} - Normalized Close Prices with Average"
            )
            plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1), fontsize=8, ncol=1)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show(block=False)
            plt.pause(3)
            plt.close()

    def demonstrateNoiseEffect(self, noises):
        horizon = 10  # time horizon to visualize over
        """
        Plot the Apple stock prices from the testing data.
        """
        plt.figure(figsize=(12, 6))
        for noiseVal in noises:
            testData = self.datasetsAndDetails["testing_data"][
                next(iter(self.datasetsAndDetails["testing_data"].keys()))
            ].copy()
            noise = np.random.normal(0, noiseVal, size=testData.shape)
            frame = testData + (noise * testData.std().values)
            closingPrices = frame["Close"].values
            limitedHorizon = max(min(horizon, len(closingPrices)), 2)
            plt.plot(closingPrices[:limitedHorizon], label=f"Noise Level = {noiseVal}")

        plt.title(
            f"{next(iter(self.datasetsAndDetails["testing_data"].keys()))} Price with Noise Levels. Horizon = {horizon}"
        )
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid()
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    def setData(self, dataType, useNoiseEval=True, epoch=0):
        marketData = self.datasetsAndDetails[f"{dataType}_data"]
        normalisedData = {}
        dataShape = next(iter(marketData.values())).shape
        PRICE_DATA = {}

        if dataType == "validation":  # noise if validation
            # Following Liang et al. (2018) - noise perturbation - "synthetic" data
            noise = None
            if useNoiseEval:
                """
                if using noise here - seed it by the epoch. This is to ensure that the noise is consistent across a given epoch for all hyperparameters
                """
                np.random.seed(self.baseSeed + epoch)
                noise = np.random.normal(0, self.perturbationNoise, size=dataShape)

        for key, dframe in marketData.items():
            df = dframe.copy()
            df.reset_index(drop=True)
            if dataType == "validation":
                # only add noise if validation data. Else (if testing) do not.
                if noise is not None:
                    df += noise * df.std().values  # make autoregressive?
            df["Return"] = df["Close"].pct_change().fillna(0)
            PRICE_DATA[key] = df["Return"].values
            normalisedData[key] = normData(
                df, windowSize=self.TIME_WINDOW, actuallyNormalise=self.normaliseData
            )  # currently not normalisiing - normalisation makes it look noisy and removes indicator information

        self.marketData = pd.DataFrame(PRICE_DATA)
        self.normDataTensors = {
            key: torch.tensor(
                normalisedData[key].values, dtype=torch.float32, device=device
            )
            for key in normalisedData.keys()
        }
        self.episodeLength = dataShape[0]

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
        :param TIME_WINDOW: The lookback window of the environment.
        :return: The observation matrix for the agent to use.
        """
        if i < TIME_WINDOW:
            raise ValueError(
                f"Current timestep i={i} is less than TIME_WINDOW={TIME_WINDOW}. Cannot extract market data."
            )

        observationMatrix = []

        if LOGGING_MARKET_DATA:
            print(f"\n=== getMarketData at timestep i = {i} ===")

        for p in range(1, TIME_WINDOW + 1):
            index = i - p + 1

            if LOGGING_MARKET_DATA:
                print(f"[p = {p}] → index = {index}")

            allocations = self.allocations[index]

            if LOGGING_MARKET_DATA:
                print(f"  Allocations[{index}] = {allocations}")

            try:
                portfolioValue = torch.tensor(
                    [self.PORTFOLIO_VALUES[index] / self.startCash],
                    dtype=torch.float32,
                    device=device,
                )
            except ZeroDivisionError:
                raise ZeroDivisionError(
                    "startCash is zero; cannot normalize portfolio value."
                )

            if LOGGING_MARKET_DATA:
                print(f"  PortfolioValue[{index}] = {portfolioValue}")

            obsVector = [allocations, portfolioValue]

            for name, dfTensor in self.normDataTensors.items():
                dataRow = dfTensor[index]
                if LOGGING_MARKET_DATA:
                    print(f"  MarketData[{name}][{index}] = {dataRow.cpu().numpy()}")

                obsVector.append(dataRow)

            try:
                combined = torch.cat(obsVector).to(device)
            except Exception as e:
                raise RuntimeError(
                    f"Error during tensor concatenation at timestep {index}: {str(e)}"
                )

            observationMatrix.insert(0, combined)

        try:
            finalMatrix = torch.stack(observationMatrix).unsqueeze(0)
        except Exception as e:
            raise RuntimeError(f"Error stacking observation matrix: {str(e)}")

        return finalMatrix

    def step(
        self, action, returnNextObs=True, observeReward=True
    ):  # if random, no need to return next obs
        newPortfolioValue = self.calculatePortfolioValue(
            action,
            self.marketData.iloc[
                self.timeStep + 1
            ].values,  # issue must be something like the data is not cached to be re-set - need to save prior env state (portfolio values, market data etc)
        )
        absolutePortfolioDifference = newPortfolioValue - self.PORTFOLIO_VALUES[-1]
        self.timeStep += 1
        info = dict()

        done = False

        if self.timeStep + 1 == self.episodeLength:
            done = True
            info["reason"] = "max_steps_reached"
        elif newPortfolioValue / self.startCash < 0.7:
            done = True
            info["reason"] = "portfolio_below_70%"

        self.PORTFOLIO_DIFFERENCES.append(absolutePortfolioDifference)
        self.PORTFOLIO_VALUES.append(newPortfolioValue)

        reward = None

        if observeReward:
            rewardMethod = self.getRewardFunction()
            if "CVaR" in rewardMethod:
                reward = self.getCVaRReward(absolutePortfolioDifference)
            elif rewardMethod == "Standard Logarithmic Returns":
                reward = self.getLogReward(
                    self.PORTFOLIO_VALUES[-1], self.PORTFOLIO_VALUES[-2]
                )  # return ln(P_t / P_t-1) = ln(1 + r)
            elif "Differential" in rewardMethod:
                reward = self.calculateDifferentialSharpeRatio(
                    self.PORTFOLIO_VALUES[-1], self.PORTFOLIO_VALUES[-2]
                )
            else:
                raise ValueError("Unknown reward method: " + rewardMethod)

        if not done and returnNextObs:
            nextObs = self.getData(self.timeStep)
        else:
            nextObs = torch.zeros(
                (1, self.TIME_WINDOW, self.numberOfFeatures),
                dtype=torch.float32,
                device=device,
            )
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
        if portfolioValues is None:
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
        if portfolioValues is None:
            portfolioValues = self.PORTFOLIO_VALUES
        maxValue = float("-inf")
        maxDrawdown = 0.0
        for value in portfolioValues:
            maxValue = max(maxValue, value)
            drawdown = (maxValue - value) / maxValue * 100
            maxDrawdown = max(maxDrawdown, drawdown)
        return maxDrawdown

    def getLogReward(self, mostRecentPortfolioValue=None, previousPortfolioValue=None):
        """
        Returns the logarithmic reward based on the portfolio values.
        :param mostRecentPortfolioValue: The most recent portfolio value.
        :param previousPortfolioValue: The previous portfolio value.
        :return: The logarithmic reward.
        return ln(P_t / P_t-1) = ln(1 + r)
        """
        if LOGGING_LOG_REWARD:
            print("+" * 50)
            print(
                f"Calculating Log Reward: mostRecentPortfolioValue = {mostRecentPortfolioValue}, previousPortfolioValue = {previousPortfolioValue}"
            )
            print(
                f"Log Reward = ln({mostRecentPortfolioValue} / {previousPortfolioValue}) = {np.log(mostRecentPortfolioValue / previousPortfolioValue) if previousPortfolioValue is not None else 0.0}"
            )
            print("+" * 50)
        actualReward = None
        try:
            actualReward = (
                np.log(mostRecentPortfolioValue / previousPortfolioValue)
                if previousPortfolioValue is not None
                else 0.0
            )
        except Exception as e:
            raise ValueError(
                f"Error calculating log reward: {e}. mostRecentPortfolioValue = {mostRecentPortfolioValue}, previousPortfolioValue = {previousPortfolioValue}"
            )
        return actualReward

    def calculateDifferentialSharpeRatio(
        self, mostRecentPortfolioValue, previousPortfolioValue
    ):
        """
        In line with Moody & Saffel's "Reinforcement Learning for Trading" 1998 Paper
        It is best to look at the formula found in this paper:
            https://papers.nips.cc/paper_files/paper/1998/file/4e6cd95227cb0c280e99a195be5f6615-Paper.pdf
        The relevant equations are found on page 3.
        """
        # initialisation
        currentReturn = mostRecentPortfolioValue / previousPortfolioValue - 1
        if self.meanReturn is None:
            self.meanReturn = currentReturn
            self.meanSquaredReturn = currentReturn**2
            return 0.0

        if LOGGING_DSR_REWARD:
            print("^" * 50)
            print(
                f"Calculating Differential Sharpe Ratio with currentReturn = {currentReturn}"
            )
        # A_t-1
        prevMeanReturn = self.meanReturn

        # B_t-1
        prevMeanSquaredReturn = self.meanSquaredReturn

        # change in each - just that delta
        deltaMean = currentReturn - prevMeanReturn
        deltaSquared = currentReturn**2 - prevMeanSquaredReturn

        # A_t, B_t
        self.meanReturn += self.decayRate * deltaMean
        self.meanSquaredReturn += self.decayRate * deltaSquared

        # denominator, straighforward
        denom = (prevMeanSquaredReturn - prevMeanReturn**2) ** 1.5
        if denom == 0:
            return 0.0

        # numerator, straightforward
        numerator = (
            prevMeanSquaredReturn * deltaMean - 0.5 * prevMeanReturn * deltaSquared
        )
        differentialSharpeRatio = numerator / denom

        if LOGGING_DSR_REWARD:
            print(f"Denominator = {denom}, Numerator = {numerator}")
            print(f"Differential Sharpe Ratio = {numerator / denom}")
            print("^" * 50)

        return differentialSharpeRatio

    def getCVaRReward(self, r, useCVaR=True):
        """
        CVaR reward calculation.
        :param r: The reward to be calculated.
        :param useCVaR: Whether to use CVaR or not. If not, the reward is just the normalised value of the reward.
        It should be logChangeInCVaR penalty, otherwise normal log returns.
        Issue: at the first timestep where CVaR is calculated, the change will be large as the previous value is 0. This isn't an issue, since the agent never sees this, but would need to be dealt with if the agent did.
        """
        if useCVaR and self.agentRiskAversion != 0:
            if LOGGING_CVAR_REWARD:
                print("*" * 50)
                print(f"Calculating CVaR Reward with r = {r}")
            currentCVaR = self.calculateCVaR()
            if LOGGING_CVAR_REWARD:
                print(f"Current CVaR = {currentCVaR}, Previous CVaR = {self.CVaR[-1]}")
            changeInCVaR = -(currentCVaR - self.CVaR[-1])
            cVaRNum = changeInCVaR
            if LOGGING_CVAR_REWARD:
                print(f"Change in CVaR = {changeInCVaR}, Normalised = {cVaRNum}")
            riskPenalty = self.agentRiskAversion * cVaRNum
            if LOGGING_CVAR_REWARD:
                print(f"Risk Penalty = {riskPenalty}")
            self.CVaR.append(currentCVaR)
        else:
            riskPenalty = 0
        scaledReward = self.getLogReward(
            self.PORTFOLIO_VALUES[-1], self.PORTFOLIO_VALUES[-2]
        )
        finalReward = scaledReward - riskPenalty
        if LOGGING_CVAR_REWARD:
            print(
                f"Scaled Reward = {scaledReward}, Final Reward (after penalty)= {finalReward}"
            )
            print("*" * 50)
        return finalReward

    def reset(
        self,
        seed=None,
        options=None,
        episode=0,
        epoch=0,
        evalType="validation",
        pushWindow=False,
    ):
        super().reset(seed=seed)
        self.timeStep = 0
        self.allocations = []
        self.previousPortfolioValue = None
        self.PORTFOLIO_VALUES = [self.startCash]
        self.isReady = False
        self.CVaR = [0]
        self.PORTFOLIO_DIFFERENCES = [0]
        self.meanReturn = None
        self.meanSquaredReturn = None
        if pushWindow:
            self.pushTrainingWindow(episode=episode, epoch=epoch, evalType=evalType)

    def calculatePortfolioValue(self, targetAllocation, closingPriceChanges):
        """
        Portfoloio Value Calculation Logic. This is detailed in section 3.1.1 of the paper.
        targetAllocation: Desired allocation for end of day.
        closingPriceChanges: Changes in the prices over the day, relative to the previous day.
        """
        if not isinstance(targetAllocation, torch.Tensor):
            targetAllocation = torch.tensor(
                targetAllocation, dtype=torch.float32, device=device
            )

        targetAllocation = targetAllocation / torch.sum(
            targetAllocation
        )  # normalise to sum to 1

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

        # effect the price changes over the day - see how each portfolio allocation changes
        wealthDistribution = self.previousPortfolioValue * currentAllocation
        changeWealth = (1 + closingPriceChanges) * wealthDistribution

        # compute transaction cost based on money held. You are penalised heavier for stronger movements
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
        Limitations with this method:
        - There may be instability in the calculation because the length of the returns array is changing - a pure limitation of the historical method (IF AND ONLY IF YOU VARY THE LENGTH OF THE RETURNS ARRAY (I don't think people do this, but I did because I wanted to get the rf involved quickly)).
        - It is either we fix the returns array length and calculate of a smaller fixed horizon, which may be unstable, or:
        - we let it get longer and more accurate.
        - but maybe this is the point - as long as the agent doesn't make decisions that result in an increase in CVaR/downside risk, then it's fine for it to be rewarded?
        """
        if len(self.CVaR) < max(10, int(1 / percentage)):
            return np.mean(self.CVaR)
        sortedReturns = sorted(self.PORTFOLIO_DIFFERENCES)
        indexToBePicked = int(np.ceil(percentage * len(sortedReturns)))
        CVaR = np.mean(sortedReturns[:indexToBePicked])
        return CVaR

    def animatePortfolio(self, save_path=None):
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 10), constrained_layout=True
        )

        categories = ["Cash"] + list(self.marketData.columns)
        numberOfAssets = len(categories)

        # Define consistent colormap
        colourMap = plt.colormaps.get_cmap("nipy_spectral")
        assetColours = [
            colourMap(i / (numberOfAssets - 1)) for i in range(numberOfAssets)
        ]

        def update(frame):
            try:
                ax1.clear()
                ax2.clear()
                ax3.clear()

                timeAxis = list(range(frame + 1))
                portfolioValues = self.PORTFOLIO_VALUES[: frame + 1]
                tensorWeights = torch.stack(self.allocations[: frame + 1]).cpu()
                weights = tensorWeights.numpy()

                # Plot 1: Portfolio value over time
                ax1.plot(
                    timeAxis, portfolioValues, label="Portfolio Value", color="blue"
                )
                ax1.set_title("Portfolio Value Over Time")
                ax1.set_ylabel("Value")
                ax1.grid(True)
                ax1.legend(loc="upper left")

                # Plot 2: Portfolio weights over time
                for i in range(weights.shape[1]):
                    ax2.plot(
                        timeAxis,
                        weights[:, i],
                        label=categories[i],
                        color=assetColours[i],
                    )
                ax2.set_title("Portfolio Weights Over Time")
                ax2.set_ylabel("Weight")
                ax2.set_xlabel("Step")

                min_val = self.allocations[frame].min().item()
                max_val = self.allocations[frame].max().item()
                ax2.set_ylim([min_val - 0.005, max_val + 0.005])
                ax2.grid(True)

                # Better legend positioning
                ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)

                # Plot 3: Current allocation as bar chart
                currentAllocation = self.allocations[frame].cpu().numpy()
                bars = ax3.bar(categories, currentAllocation, color=assetColours)

                ax3.set_xlabel("Assets")
                ax3.set_ylabel("Proportion Allocated")
                ax3.set_title(f"Allocations at Step {frame}")
                ax3.tick_params(axis="x", labelsize=8)
                ax3.set_xticks(range(len(categories)))
                ax3.set_xticklabels(categories, rotation=45, ha="right")

                # Custom legend only once (colors match lines and bars)
                if frame == 0:  # Only add legend once to avoid clutter
                    ax3.legend(
                        bars,
                        categories,
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1),
                        fontsize=7,
                    )

            except Exception as e:
                print(f"Error at frame {frame}: {e}")
                raise e

        ani = FuncAnimation(
            fig, update, frames=len(self.PORTFOLIO_VALUES), repeat=False
        )

        writer = FFMpegWriter(fps=10, metadata=dict(artist="Richard"), bitrate=1800)
        ani.save(save_path, writer=writer, dpi=150)
        print(f"Animation saved to {save_path}")
        plt.close(fig)

    def warmUp(self, observeReward=True):
        """
        'warm up' environment until there's enough data to estimate CVaR
        and to create a long enough time window to pass to the lstm.
        """
        for _ in range(self.TIME_WINDOW):
            self.step(
                np.ones(self.numberOfAssets + 1) / (self.numberOfAssets + 1),
                observeReward=observeReward,
                returnNextObs=False,
            )
        self.setIsReady(True)

    def pushTrainingWindow(self, episode, epoch, evalType):
        """
        Training environment initialization function. From the base training data, we
        generate a slightly perturbed version of the orignal training data, ensuring
        that the noise added is dependent on both the episode and the epoch.
        This ensures that no two episodes out of a whole training run are the same, even
        if they cover the same base periods from the initial training data.
        """
        start = self.datasetsAndDetails["timestep_shift"] * episode

        datasets = {
            "validation": {
                "TRAINING_DATA": self.datasetsAndDetails["training_data"],
            },
            "testing": {  # use traingin and validation data combo for testing
                "TRAINING_DATA": self.datasetsAndDetails["training_validation_data"],
            },
        }
        DATA = datasets[evalType]["TRAINING_DATA"]
        dataShape = list(DATA.values())[0].shape
        end = min(start + self.datasetsAndDetails["episode_length"], dataShape[0])
        dataWindow = {}
        for key, value in DATA.items():
            dataWindow[key] = value.iloc[start:end].copy()

        np.random.seed(
            self.baseSeed + episode * 100 + epoch
        )  # each episode has a different seed
        NOISY_PRICE_DATA = {}
        normalisedData = {}
        for key, dframe in dataWindow.items():
            df = dframe.copy()
            df = df.reset_index(drop=True)
            # Following Liang et al. (2018) - noise perturbation - "synthetic" data
            noise = (
                np.random.normal(
                    0, self.perturbationNoise, size=(end - start, dataShape[1])
                )
                * df.std().values
            )
            df += noise  # make autoregressive?
            df["Return"] = df["Close"].pct_change().fillna(0)
            NOISY_PRICE_DATA[key] = df["Return"].values
            normalisedData[key] = normData(
                df, windowSize=self.TIME_WINDOW, actuallyNormalise=self.normaliseData
            )
        self.marketData = pd.DataFrame(NOISY_PRICE_DATA)
        self.normDataTensors = {
            key: torch.tensor(df.values, dtype=torch.float32, device=device)
            for key, df in normalisedData.items()
        }
        # between the current and previous time step.
        self.episodeLength = self.datasetsAndDetails["episode_length"]

    def getIsReady(self):
        """
        These let the training loop know when the environment is ready to be used.
        """
        return self.isReady

    def setIsReady(self, boolean: bool):
        self.isReady = boolean

    def setRewardFunction(self, rewardFunction):
        """
        Set the reward function to be used in the environment.
        :param rewardFunction: The reward function to be used.
        """
        if "Differential" in rewardFunction:
            decay = float(rewardFunction.split("_")[1])
            self.decayRate = decay
        self.rewardFunction = rewardFunction

    def getRewardFunction(self):
        """
        Get the reward function to be used in the environment.
        :return: The reward function to be used.
        """
        return self.rewardFunction

    def generateAnimation(self, agentType, stage, index):
        higherDir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        file = f"{higherDir}/animations/{index}/{agentType}/{stage}/"
        if not os.path.exists(file):
            os.makedirs(file)
        self.animatePortfolio(f"{file}/portfolio_animation.mp4")
