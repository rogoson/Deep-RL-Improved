# import pytest
# from main.environments.TimeSeriesEnvironment import TimeSeriesEnvironment
# from main.experiments.InitialisationHelpers import getEnv
# from scipy.special import softmax
# import pandas as pd
# import yaml
# import torch
# import numpy as np
# from pprint import pprint
# import os

# configPath = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml")
# )
# device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

# with open(configPath) as file:
#     yamlConfiguration = yaml.safe_load(file)


# def test_env_setup():

#     testEnv = getEnv(yamlConfiguration)
#     setupInformation = testEnv.setup(yamlConfiguration)

#     envVariables = vars(testEnv)
#     pprint(envVariables)
#     pprint(setupInformation)


# def test_set_out_of_sample_validation():
#     testEnv = getEnv(yamlConfiguration)
#     _ = testEnv.setup(yamlConfiguration)

#     testEnv.setData("validation")

#     envVariables = vars(testEnv)
#     pprint(envVariables)


# def test_set_out_of_sample_testing():
#     testEnv = getEnv(yamlConfiguration)
#     _ = testEnv.setup(yamlConfiguration)

#     testEnv.setData("testing")

#     envVariables = vars(testEnv)
#     pprint(envVariables)


# def test_reset():
#     testEnv = getEnv(yamlConfiguration)
#     _ = testEnv.setup(yamlConfiguration)

#     testEnv.reset()
#     envVariables = vars(testEnv)
#     pprint(envVariables)


# def test_get_market_data_shape():
#     testEnv = getEnv(yamlConfiguration)
#     setupInformation = testEnv.setup(yamlConfiguration)
#     timeWindow = yamlConfiguration["time_window"]

#     testEnv.reset()
#     testEnv.setRewardFunction("Standard Logarithmic Returns")

#     for step in range(timeWindow):
#         _ = testEnv.step(
#             action=softmax(
#                 np.random.normal(scale=1, size=setupInformation["numberOfAssets"] + 1)
#             ),
#             returnNextObs=False,
#         )
#     output = testEnv.getData()

#     assert isinstance(output, torch.Tensor)
#     # Dimensions should be: (1, TIME_WINDOW, total_features)
#     assert output.shape[0] == 1
#     assert output.shape[1] == timeWindow

#     # Expected feature vector length:
#     expected_feat_dim = len(testEnv.allocations[0]) + 1  # +1 for portfolioValue
#     for tensor in testEnv.normDataTensors.values():
#         expected_feat_dim += tensor.shape[1]

#     assert output.shape[2] == expected_feat_dim


# def test_index_consistency():
#     i = 37
#     testEnv = getEnv(yamlConfiguration)
#     setupInfo = testEnv.setup(yamlConfiguration)

#     testEnv.reset()
#     testEnv.setRewardFunction("Standard Logarithmic Returns")

#     for step in range(i):
#         _ = testEnv.step(
#             action=softmax(
#                 np.random.normal(scale=1, size=setupInfo["numberOfAssets"] + 1)
#             ),
#             returnNextObs=False,
#         )
#     result = testEnv.getData(timeStep=i)

#     # Oldest timestep in the observation window
#     expected_index = i - testEnv.TIME_WINDOW + 1
#     actual_vector = result.squeeze(0)[0]

#     #  Validate allocations
#     expected_alloc = testEnv.allocations[expected_index]
#     alloc_len = expected_alloc.shape[0]
#     actual_alloc = actual_vector[:alloc_len]

#     assert torch.allclose(
#         expected_alloc, actual_alloc, atol=1e-6
#     ), "Allocations mismatch"

#     #  Validate portfolio value
#     expected_pv = torch.tensor(
#         [testEnv.PORTFOLIO_VALUES[expected_index] / testEnv.startCash],
#         dtype=torch.float32,
#         device=device,
#     )
#     actual_pv = actual_vector[alloc_len : alloc_len + 1]

#     assert torch.allclose(expected_pv, actual_pv, atol=1e-6), "Portfolio value mismatch"

#     #  Validate each market feature
#     start = alloc_len + 1
#     for name, dataTensor in testEnv.normDataTensors.items():
#         feat_len = dataTensor.shape[1]
#         expected_feature = dataTensor[expected_index]
#         actual_feature = actual_vector[start : start + feat_len]

#         assert torch.allclose(
#             expected_feature, actual_feature, atol=1e-6
#         ), f"Market data mismatch in '{name}'"

#         start += feat_len


# def getTestEnv(steps=None):
#     env = getEnv(yamlConfiguration)
#     env.hasVisualised = True
#     setupInfo = env.setup(yamlConfiguration)  # some of these need step logic
#     env.reset()
#     env.setRewardFunction("Standard Logarithmic Returns")

#     if steps is not None:
#         for step in range(steps):
#             _ = env.step(
#                 action=softmax(
#                     np.random.normal(scale=1, size=setupInfo["numberOfAssets"] + 1)
#                 ),
#                 returnNextObs=False,
#             )

#     return env


# def test_raises_value_error_for_low_timestep():
#     env = getTestEnv()
#     with pytest.raises(ValueError, match="Cannot extract market data"):
#         env.getMarketData(i=env.TIME_WINDOW - 1, TIME_WINDOW=env.TIME_WINDOW)


# def test_raises_zero_division_error_for_start_cash_zero():
#     stepsToRun = yamlConfiguration["time_window"]
#     env = getTestEnv(stepsToRun)
#     env.startCash = 0.0
#     with pytest.raises(ZeroDivisionError, match="startCash is zero"):
#         env.getMarketData(i=stepsToRun, TIME_WINDOW=env.TIME_WINDOW)


# def test_raises_runtime_error_for_concat_failure():
#     stepsToRun = yamlConfiguration["time_window"] + 10
#     env = getTestEnv(stepsToRun)
#     target_idx = stepsToRun - env.TIME_WINDOW + 1
#     env.allocations[target_idx] = torch.randn(99)
#     with pytest.raises(RuntimeError, match="tensor concatenation"):
#         randomIndex = np.random.randint(
#             1, stepsToRun - yamlConfiguration["time_window"]
#         )
#         env.getMarketData(i=stepsToRun - randomIndex, TIME_WINDOW=env.TIME_WINDOW)


# def test_raises_runtime_error_for_stack_failure():
#     stepsToRun = yamlConfiguration["time_window"] + 12
#     env = getTestEnv()

#     # Replace method to force a stack failure
#     def broken_getMarketData(i, TIME_WINDOW):
#         obsMatrix = [torch.randn(3), torch.randn(4)]
#         try:
#             return torch.stack(obsMatrix).unsqueeze(0)
#         except Exception as e:
#             raise RuntimeError("Error stacking observation matrix: " + str(e))

#     env.getMarketData = broken_getMarketData
#     with pytest.raises(RuntimeError, match="stacking observation matrix"):
#         env.getMarketData(i=stepsToRun, TIME_WINDOW=env.TIME_WINDOW)


# def test_get_log_reward():
#     prevPortfolioValue = 10000
#     nextPortfolioValue = 12000

#     env = getEnv(yamlConfiguration)
#     env.hasVisualised = True

#     assert (
#         np.exp(env.getLogReward(nextPortfolioValue, prevPortfolioValue))
#         * prevPortfolioValue
#     ) == nextPortfolioValue


# @pytest.fixture
# def env():
#     """
#     Instantiate a fresh TimeSeriesEnvironment with zero steps taken,
#     then override the financial parameters and reset its internal state
#     for portfolio‐value testing.
#     """
#     e = getTestEnv()

#     # override for deterministic tests
#     e.startCash = 1000.0
#     e.maxAllocationChange = 1.0
#     e.transactionCost = 0.0

#     # reset state used by calculatePortfolioValue
#     e.previousPortfolioValue = None
#     e.allocations = []
#     return e


# def test_initial_full_rebalance_no_cost(env):
#     """
#     First invocation, full rebalance to [0.5, 0.5], price change 20% on asset:
#       cash stays 0.5*1000=500 → 500*(1+0)=500
#       asset 0.5*1000=500 → 500*(1+0.2)=600
#       total = 1100
#     """
#     val = env.calculatePortfolioValue([0.5, 0.5], [0.2])
#     assert pytest.approx(1100.0, rel=1e-6) == val


# def test_subsequent_calls_accumulate_returns(env):
#     """
#     Two sequential full‐rebalance calls:
#       1) 1000 → 1100 (as above)
#       2) Starting 1100, same alloc & same 20% change:
#          1100*0.5*1 + 1100*0.5*1.2 = 550 + 660 = 1210
#     """
#     first = env.calculatePortfolioValue([0.5, 0.5], [0.2])
#     second = env.calculatePortfolioValue([0.5, 0.5], [0.2])

#     assert pytest.approx(1100.0, rel=1e-6) == first
#     assert pytest.approx(1210.0, rel=1e-6) == second


# def test_normalization_of_inputs(env):
#     """
#     Target allocation [2,2] should normalize to [0.5,0.5].
#     With zero price movement and full rebalance, portfolio stays at 1000.
#     """
#     val = env.calculatePortfolioValue([2, 2], [0.0])
#     assert pytest.approx(1000.0, rel=1e-6) == val


# def test_transaction_cost_and_partial_rebalance(env):
#     """
#     Override settings: partial rebalance (max change=0.5) and nonzero cost.
#     1st call → all cash stays 1000.
#     2nd call target flips to 100% asset:
#       prevAlloc=[1,0], target=[0,1]
#       currAlloc=0.5*prev + 0.5*target = [0.5,0.5]
#       wealthDistribution = 1000*[0.5,0.5] = [500,500]
#       no price moves → sum(changeWealth)=1000
#       txnCost = 0.02 * 1000 * (|0.5-1| + |0.5-0|)
#               = 0.02 * 1000 * (0.5 + 0.5) = 20
#       final = 1000 - 20 = 980
#     """
#     # adjust env parameters
#     env.startCash = 1000.0
#     env.maxAllocationChange = 0.5
#     env.transactionCost = 0.02

#     first = env.calculatePortfolioValue([1.0, 0.0], [0.0])
#     assert pytest.approx(1000.0, rel=1e-6) == first

#     second = env.calculatePortfolioValue([0.0, 1.0], [0.0])
#     assert pytest.approx(980.0, rel=1e-6) == second


# def test_tensor_and_list_inputs_equivalence(env):
#     """
#     Passing torch.Tensor vs Python list yields the same result.
#     """
#     tensor_val = env.calculatePortfolioValue(
#         torch.tensor([0.3, 0.7], dtype=torch.float32, device=device),
#         torch.tensor([0.05], dtype=torch.float32, device=device),
#     )

#     # reset state so we start from scratch again
#     env.previousPortfolioValue = None
#     env.allocations = []

#     list_val = env.calculatePortfolioValue([0.3, 0.7], [0.05])

#     assert pytest.approx(list_val, rel=1e-6) == tensor_val


# def test_shape_and_columns_validation(env):
#     """
#     For evalType="validation" and episode=0, epoch=0:
#     - marketData has shape (episode_length, num_assets)
#     - columns match asset keys in training_data
#     """
#     ds = env.datasetsAndDetails
#     asset_keys = set(ds["training_data"].keys())
#     L = ds["episode_length"]

#     env.pushTrainingWindow(episode=0, epoch=0, evalType="validation")

#     # rows = episode_length, cols = number of assets
#     assert env.marketData.shape == (L, len(asset_keys))
#     assert set(env.marketData.columns) == asset_keys


# def test_return_values_episode0(env):
#     """
#     Verify the computed Returns for episode=0:
#     raw_close = training_data[asset]["Close"]
#     returns = [0, Δ1/1, Δ2/2, ...]
#     """
#     ds = env.datasetsAndDetails
#     asset = next(iter(ds["training_data"]))
#     close_series = ds["training_data"][asset]["Close"].values

#     L = ds["episode_length"]
#     env.pushTrainingWindow(episode=0, epoch=0, evalType="validation")

#     actual = env.marketData[asset].values
#     window = close_series[:L]
#     expected = np.empty(L, dtype=float)
#     expected[0] = 0.0
#     expected[1:] = (window[1:] - window[:-1]) / window[:-1]

#     np.testing.assert_allclose(actual, expected, atol=1e-8)


# def test_reproducible_noise(env):
#     """
#     If perturbationNoise > 0, pushing the same episode/epoch twice
#     yields identical marketData (due to fixed seed logic).
#     """
#     env.perturbationNoise = 0.5

#     env.pushTrainingWindow(episode=2, epoch=7, evalType="validation")
#     first = env.marketData.copy()

#     env.pushTrainingWindow(episode=2, epoch=7, evalType="validation")
#     second = env.marketData.copy()

#     pd.testing.assert_frame_equal(first, second)
