# DRY to the max
from main.utils.RestServer import startServer
from main.utils.VolumeWriter import copyOver
import os
import yaml
from pathlib import Path


def ensureMarketConfig(market):
    """
    Returns the path to configs/temp_config_{market}.yaml, creating it from configs/config.yaml
    if it is missing, so the container never depends on the host having generated it.
    """
    configDir = Path(__file__).resolve().parent.parent.parent.parent / "configs"
    configPath = configDir / f"temp_config_{market}.yaml"
    if not configPath.exists():
        with open(configDir / "config.yaml") as file:
            config = yaml.safe_load(file)
        config["active_index"] = market
        with open(configPath, "w") as file:
            yaml.dump(config, file)
        print(f"Created {configPath.name} from config.yaml")
    return configPath


def applyQuickTestOverrides(yamlConfig):
    """
    Shrinks a run down to minutes so the whole train -> plot -> copy path can be checked
    before committing to a multi-week run. Enabled with the QUICK_TEST=1 env var.
    """
    # 2 seeds (not 1) so the cross-seed aggregation code is exercised too
    yamlConfig["varied_base_seeds"] = yamlConfig["varied_base_seeds"][:2]
    yamlConfig["epochs"] = 1
    yamlConfig["wandb_state"] = "disabled"
    yamlConfig["test"]["learning_curve_frequency"] = 250
    yamlConfig["max_episodes_per_epoch"] = 2
    yamlConfig["random_repeats"] = 20
    yamlConfig["reward_subset"] = [  # one of each reward family
        "Standard Logarithmic Returns",
        "Differential Sharpe Ratio_0.05",
        "CVaR_0.25",
    ]
    print("QUICK_TEST enabled - results are NOT meaningful, only checking the pipeline.")


def runExperimentFunction(experimentFunction):
    market = os.environ["MARKET_INDEX"]
    configPath = ensureMarketConfig(market)
    with open(configPath) as file:
        yamlConfiguration = yaml.safe_load(file)

    quickTest = os.environ.get("QUICK_TEST") == "1"
    if quickTest:
        applyQuickTestOverrides(yamlConfiguration)

    startServer()
    originalEpochs = yamlConfiguration["epochs"]
    td3Epochs = 1  # manualOverride, not great but we need to finish this
    print(f"Original Epochs: {originalEpochs}.")
    try:
        yamlConfiguration["epochs"] = td3Epochs
        experimentFunction(yamlConfig=yamlConfiguration, agentType="td3")
        copyOver()  # checkpoint td3 results before the (long) ppo run
        yamlConfiguration["epochs"] = originalEpochs
        experimentFunction(yamlConfig=yamlConfiguration, agentType="ppo")
    finally:
        copyOver()  # --rm containers lose everything not on the volume
