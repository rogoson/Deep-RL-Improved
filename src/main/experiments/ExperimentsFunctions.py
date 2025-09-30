# DRY to the max
from main.utils.RestServer import startServer
from main.utils.VolumeWriter import copyOver
import os
import yaml


def runExperimentFunction(experimentFunction):
    configPath = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "data",
            "configs",
            "config.yaml",
        )
    )
    with open(configPath) as file:
        yamlConfiguration = yaml.safe_load(file)

    startServer()
    originalEpochs = yamlConfiguration["epochs"]
    td3Epochs = 2  # manualOverride
    yamlConfiguration["epochs"] = td3Epochs
    experimentFunction(yamlConfig=yamlConfiguration, agentType="td3")
    yamlConfiguration["epochs"] = originalEpochs
    experimentFunction(yamlConfig=yamlConfiguration, agentType="ppo")
    copyOver()
