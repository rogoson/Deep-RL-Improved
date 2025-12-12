# DRY to the max
from main.utils.RestServer import startServer
from main.utils.VolumeWriter import copyOver
import os
import yaml
from pathlib import Path


def runExperimentFunction(experimentFunction):
    market = os.environ["MARKET_INDEX"]
    configPath = (
        Path(__file__).resolve().parent.parent.parent.parent
        / f"configs/temp_config_{market}.yaml"
    )
    with open(configPath) as file:
        yamlConfiguration = yaml.safe_load(file)

    startServer()
    originalEpochs = yamlConfiguration["epochs"]
    td3Epochs = 1  # manualOverride, not great but we need to finish this
    print(f"Original Epochs: {originalEpochs}.")
    yamlConfiguration["epochs"] = td3Epochs
    experimentFunction(yamlConfig=yamlConfiguration, agentType="td3")
    yamlConfiguration["epochs"] = originalEpochs
    experimentFunction(yamlConfig=yamlConfiguration, agentType="ppo")
    copyOver()
