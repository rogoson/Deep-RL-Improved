from main.utils.GeneralUtils import seed, getFileWritingLocation
from main.trainingAndEval.Training import trainingLoop
from .InitialisationHelpers import getEnv
from .ExperimentsFunctions import runExperimentFunction
from .NonTestExperimentsPlotting import (
    runParameterComparison,
)
import os
import wandb
import numpy as np


def hyperparameterTuning(yamlConfig, agentType="ppo", phase="hyperparameter_tuning"):
    """
    Runs the hyperparameter sweep by sequentially activating one test type at a time.
    """
    TESTING = {
        "FEATURE OUTPUT SIZE": False,
        # "NORMALIZE DATA": False,
    }
    portfolioValues = dict()
    for extractor in [True, False]:
        yamlConfig["usingLSTMFeatureExtractor"] = extractor
        for key in list(TESTING.keys()):
            print("=" * 50)
            TESTING[key] = True

            sweepParams = {
                "FEATURE OUTPUT SIZE": {
                    "values": yamlConfig["hyperparameters"]["feature_output_sizes"],
                    "overrides": {"feature_output_size": None},
                },
                # "NORMALIZE DATA": {  # not really a hyperparam but easier than a whole other experiment
                #     "values": [True, False],
                #     "overrides": {},
                # },
            }

            # Iterate over active test types defined in TESTING
            for s in yamlConfig["varied_base_seeds"]:
                BASE_SEED = s
                seed(BASE_SEED)
                yamlConfig["env"]["base_seed"] = s
                for testType, active in TESTING.items():
                    if active and testType in sweepParams:
                        param_info = sweepParams[testType]
                        for value in param_info["values"]:
                            print(f"Running sweep for {testType}: {value}")
                            # Build the overrides dict, substituting sweep values where needed.
                            overrides = {
                                key: (value if override is None else override)
                                for key, override in param_info["overrides"].items()
                            }
                            if testType == "NORMALIZE DATA":
                                yamlConfig["normalise_data"] = value
                            env = getEnv(yamlConfig)
                            portfolioValues[s] = trainingLoop(
                                yamlConfig,
                                env,
                                agentType,
                                stage=phase,
                                conf=f"{testType.lower().title()} - {value} | Strategy-{agentType}",
                                optionalHyperConfig=overrides,
                            )
                            baseFolder = getFileWritingLocation(
                                yamlConfig, agentType=agentType
                            )
                            desiredFolder = (
                                f"{baseFolder}/portfolios/{testType}/{value}/{s}/"
                            )
                            if not os.path.exists(desiredFolder):
                                os.makedirs(desiredFolder)
                            np.savetxt(
                                f"{desiredFolder}validationPerformances.txt",
                                portfolioValues[s]["validation_performances"],
                                fmt="%f",
                            )
                            wandb.finish()
                        break  # Run only one active test type per sweep
            TESTING[key] = False
            print("=" * 50)
    runParameterComparison(yamlConfig, env, agentType)


if __name__ == "__main__":
    runExperimentFunction(hyperparameterTuning)
