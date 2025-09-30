import main.experiments.RewardTesting as rewTesting
import main.experiments.HyperparameterTuning as hypTuning
from main.utils.RestServer import startServer
import pytest
import os
import torch
import yaml


configPath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml")
)

with open(configPath) as file:
    yamlConfiguration = yaml.safe_load(file)
yamlConfiguration["epochs"] = 1
yamlConfiguration["varied_base_seeds"] = [1, 2]
yamlConfiguration["source_folder"] = "tests"


def test_hyperparameterTuning():
    startServer()
    hypTuning.hyperparameterTuning(
        yamlConfig=yamlConfiguration
    )  # needs work on saving models
    hypTuning.hyperparameterTuning(yamlConfig=yamlConfiguration, agentType="td3")


def test_actualTesting():
    startServer()
    rewTesting.trainTestingAgents(yamlConfig=yamlConfiguration)
    rewTesting.trainTestingAgents(yamlConfig=yamlConfiguration, agentType="td3")


yamlConfiguration["usingLSTMFeatureExtractor"] = True
test_actualTesting()
