import main.experiments.ExperimentsFunctions as mainExp
import pytest
import os
import torch
import yaml


configPath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml")
)
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

with open(configPath) as file:
    yamlConfiguration = yaml.safe_load(file)
yamlConfiguration["epochs"] = 1
yamlConfiguration["varied_base_seeds"] = [1, 2]
yamlConfiguration["source_folder"] = "tests"


def test_normalisationEffectExperiment():
    mainExp.normalisationEffectExperiment(yamlConfig=yamlConfiguration, agentType="td3")
    mainExp.normalisationEffectExperiment(yamlConfig=yamlConfiguration, agentType="ppo")


def test_noiseTestingExperiment():
    mainExp.noiseTestingExperiment(yamlConfig=yamlConfiguration)
    mainExp.noiseTestingExperiment(yamlConfig=yamlConfiguration, agentType="td3")


def test_hyperparameterTuning():
    mainExp.hyperparameterTuning(
        yamlConfig=yamlConfiguration
    )  # needs work on saving models
    mainExp.hyperparameterTuning(yamlConfig=yamlConfiguration, agentType="td3")


def test_actualTesting():
    # mainExp.trainTestingAgents(yamlConfig=yamlConfiguration)
    mainExp.trainTestingAgents(yamlConfig=yamlConfiguration, agentType="td3")


test_actualTesting()
