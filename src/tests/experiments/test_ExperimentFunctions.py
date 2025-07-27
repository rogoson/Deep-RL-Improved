import main.experiments.ExperimentsFunctions as mainExp
import pytest
import os
import torch
import yaml


configPath = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml")
)
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
SOURCE_FOLDER = "tests"

with open(configPath) as file:
    yamlConfiguration = yaml.safe_load(file)
yamlConfiguration["epochs"] = 1
yamlConfiguration["varied_base_seeds"] = [1, 2]


def test_normalisationEffectExperiment():
    mainExp.normalisationEffectExperiment(
        yamlConfig=yamlConfiguration, sourceFolder=SOURCE_FOLDER
    )


# fix test dir for evaluation for logging output
