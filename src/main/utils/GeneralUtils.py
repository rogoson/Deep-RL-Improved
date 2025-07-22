import numpy as np
import torch
import os


def seed(seed):
    """
    General Seed function. Called at the start of each training run.
    Ensures that all agents are initialised with the same weights.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for GPU


def normData(df, windowSize=0, actuallyNormalise=True):
    """
    Use Rolling zscore to normalise - might alter the meanings of indicators - LIMITATION
    1. Soleymani's would create temporal distortion (feature_i/feature_i-1) -
    2. Liang et al. (2018) would create temporal leakage
    Ultimately, no normalisation method is perfect, so I will have to deal with this.
    TIME_WINDOW decision is brute.
    """
    # rolling mean normalisation
    # min periods = 1 may be a limitation
    if actuallyNormalise:
        rollingDf = df.rolling(window=windowSize, min_periods=1).mean()
        rollingStd = df.rolling(
            window=windowSize, min_periods=1
        ).std()  # usage of 1 is questionable but necesary

        rollingZScoreDf = (df - rollingDf) / (rollingStd + 1e-8)

        rollingZScoreDf.columns = df.columns
        return rollingZScoreDf  # temporarily removing normalisation!!! #rollingZScoreDf.fillna(0) # fill NaNs with 0s, forced
    df = df.drop(columns=["Return"])  # remove return column if it exists
    return df


def getFileWritingLocation(directory="main"):
    fileWritingLocation = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "plotsAndPorftolioTrajectories")
    )
    if directory != "main":
        fileWritingLocation = fileWritingLocation.replace("main", "test")
    os.makedirs(fileWritingLocation, exist_ok=True)
