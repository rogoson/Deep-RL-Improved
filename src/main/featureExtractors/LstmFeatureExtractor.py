import torch
import numpy as np
from torch.nn import Linear, LSTM, ReLU, Module, init

_SAVE_SUFFIX = "_lstm"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")


def layerInit(layer, std=np.sqrt(2), biasConst=0.0):
    """
    Function taken from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """
    init.orthogonal_(layer.weight, std)
    init.constant_(layer.bias, biasConst)
    return layer


def initialiseForgetBias(
    lstm: torch.nn.LSTM, forgetBias: float = 1.0
):  # forget less! (initially)
    H = lstm.hidden_size
    suffix = "_l0"
    for biasName in [f"bias_ih{suffix}", f"bias_hh{suffix}"]:
        b = getattr(lstm, biasName)
        # [i, f, g, o], forget slice is from [H:2H]
        b.data[H : 2 * H].fill_(forgetBias)


class LstmFeatureExtractor(Module):
    """
    LSTM Feature Extractor for Agent
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
        returnHiddenState=False,
        modelName="featureExtractor",
    ):
        super(LstmFeatureExtractor, self).__init__()

        self.lstmHiddenSize = lstmHiddenSize
        self.lstmOutputSize = lstmOutputSize
        self.modelName = modelName
        self.save_file_name = self.modelName + _SAVE_SUFFIX
        self.returnHiddenState = returnHiddenState

        self.featureLSTM = LSTM(
            input_size=numFeatures,
            hidden_size=lstmHiddenSize,
            num_layers=1,
            batch_first=True,
            device=device,
        )
        initialiseForgetBias(self.featureLSTM, forgetBias=2.0)
        self.featureExtractorfc1 = layerInit(
            Linear(lstmHiddenSize, lstmHiddenSize, device=device)
        )
        self.featureExtractorfc2 = layerInit(
            Linear(lstmHiddenSize, lstmHiddenSize, device=device)
        )
        self.featureExtractorfc3 = layerInit(
            Linear(lstmHiddenSize, lstmOutputSize, device=device)
        )
        self.postLSTMNorm = torch.nn.LayerNorm(
            self.lstmHiddenSize, device=device
        )  # might normalise but could be similar problem
        self.relu = ReLU()

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
        self.featureLSTM.flatten_parameters()
        _, (hidden, cell) = self.featureLSTM(x, hiddenState)
        hiddenNormed = self.postLSTMNorm(hidden[-1])
        features = self.relu(self.featureExtractorfc1(hiddenNormed))
        features = self.relu(self.featureExtractorfc2(features))
        features = self.featureExtractorfc3(features)
        return features, ((hidden, cell) if self.returnHiddenState else hiddenState)

    def initHidden(self, batchSize=1):
        # reset hidden states
        h0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        c0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        return (h0, c0)
