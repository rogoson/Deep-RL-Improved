# DRY to the max
from main.utils.RestServer import startServer
from main.utils.VolumeWriter import copyOver
import os
import yaml


def runExperimentFunction(experimentFunction, yamlConfiguration):
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
    experimentFunction(yamlConfig=yamlConfiguration, agentType="ppo")
    experimentFunction(yamlConfig=yamlConfiguration, agentType="td3")
    copyOver()
