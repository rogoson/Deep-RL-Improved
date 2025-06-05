import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from utils import pathJoin
from torch.nn import Linear, LSTM, Tanh
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

_SAVE_SUFFIX = "_lstm"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
RETURN_HIDDEN_STATE = False


class LstmFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM Feature Extractor for PPO Agent
    This class is used to extract features from the input data using an LSTM network, and some
    linear layers as specified in the report

    This was done from scratch, but following the guidance of  stable-baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """

    def __init__(
        self,
        numFeatures,
        lstmHiddenSize=128,
        lstmOutputSize=128,
        modelName="featureExtractor",
    ):
        super(LstmFeatureExtractor, self).__init__(
            observation_space=None, features_dim=lstmOutputSize
        )
        self.lstmHiddenSize = lstmHiddenSize
        self.lstmOutputSize = lstmOutputSize
        self.modelName = modelName
        self.save_file_name = self.modelName + _SAVE_SUFFIX

        self.featureLSTM = LSTM(
            input_size=numFeatures,
            hidden_size=lstmHiddenSize,
            num_layers=1,
            batch_first=True,
        )
        self.featureExtractorfc1 = Linear(lstmHiddenSize, lstmHiddenSize)
        self.featureExtractorfc2 = Linear(lstmHiddenSize, lstmHiddenSize)
        self.featureExtractorfc3 = Linear(lstmHiddenSize, lstmOutputSize)
        self.tanh = Tanh()

    def forward(self, x, hiddenState=None):
        """
        Forward method. batch size is always 1 when processing market data, otherwise it's the size
        of the batch in the stored memory - hence (batchSize=x.size(0)).
        :param x: input data
        :param hiddenState: hidden state of the LSTM
        :return: features and hidden state
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if hiddenState is None:
            hiddenState = self.initHidden(batchSize=x.size(0))
        _, (hidden, cell) = self.featureLSTM(x, hiddenState)
        features = self.tanh(self.featureExtractorfc1(hidden[-1]))
        features = self.tanh(self.featureExtractorfc2(features))
        features = self.tanh(self.featureExtractorfc3(features))
        return features, ((hidden, cell) if RETURN_HIDDEN_STATE else hiddenState)

    def initHidden(self, batchSize=1):
        # reset hidden states
        h0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        c0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        return (h0, c0)
