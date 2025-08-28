import torch
from torch.nn import (
    Linear,
    Conv1d,
    ReLU,
    BatchNorm1d,
    MaxPool1d,
    AdaptiveAvgPool1d,
    Sequential,
    Module,
)

_SAVE_SUFFIX = "_cnn"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")


class CNNFeatureExtractor(Module):
    """
    CNN Feature Extractor for Agent
    This class is used to extract features from the input data using an AlexNet-informed CNN architecture,
    and some linear layers.
    based on https://stackoverflow.com/questions/62372938/understanding-input-shape-to-pytorch-conv1d
    and https://www.digitalocean.com/community/tutorials/writing-cnns-from-scratch-in-pytorch
    """

    def __init__(
        self, numFeatures, timeWindow, outputSize, modelName="featureExtractor"
    ):
        super(CNNFeatureExtractor, self).__init__()

        self.channels = numFeatures
        self.length = timeWindow
        self.outputSize = outputSize
        self.modelName = modelName
        self.save_file_name = self.modelName + _SAVE_SUFFIX

        self.designDecisions = {
            "firstConvOutputChannels": self.outputSize // 4,
            "firstConvPadding": 1,
            "firstConvStride": 2,
            "firstConvKernelSize": self.length // 10,
            "secondConvStride": 1,
            "secondConvKernelSize": 2,
            "secondConvOutputChannels": self.outputSize // 2,
            "thirdConvOutputChannels": self.outputSize,
        }
        self.maxPoolingParameters = {"kernelSize": 2, "stride": 2}

        self.convolutionalSection = Sequential(
            Conv1d(
                in_channels=self.channels,
                out_channels=self.designDecisions["firstConvOutputChannels"],
                kernel_size=self.designDecisions["firstConvKernelSize"],
                stride=self.designDecisions["firstConvStride"],
                padding=self.designDecisions["firstConvPadding"],
            ),
            BatchNorm1d(num_features=self.designDecisions["firstConvOutputChannels"]),
            ReLU(),
            MaxPool1d(
                kernel_size=self.maxPoolingParameters["kernelSize"],
                stride=self.maxPoolingParameters["stride"],
            ),
            Conv1d(
                in_channels=self.designDecisions[
                    "firstConvOutputChannels"
                ],  # no padding
                out_channels=self.designDecisions["secondConvOutputChannels"],
                kernel_size=self.designDecisions["secondConvKernelSize"],
                stride=self.designDecisions["secondConvStride"],
            ),
            BatchNorm1d(num_features=self.designDecisions["secondConvOutputChannels"]),
            ReLU(),
            Conv1d(
                in_channels=self.designDecisions["secondConvOutputChannels"],
                out_channels=self.designDecisions["thirdConvOutputChannels"],
                kernel_size=self.designDecisions["firstConvKernelSize"],
                stride=self.designDecisions["secondConvStride"],
                padding=self.designDecisions["firstConvPadding"],
            ),
            BatchNorm1d(num_features=self.designDecisions["thirdConvOutputChannels"]),
            ReLU(),
            AdaptiveAvgPool1d(1),  # i aint doin this myself
        )
        self.convolutionalSection = self.convolutionalSection.to("cuda")

        self.fc1 = Linear(self.outputSize, self.outputSize, device=device)

    def forward(
        self, x, hiddenStates=None
    ):  # hidden states not used in CNN, just for limited editing in agent classes - always none here
        conv = self.convolutionalSection(x.transpose(-2, -1))
        out = self.fc1(conv.squeeze(-1))
        return out, hiddenStates
