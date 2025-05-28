# Based largely on code from Phil Tabor's tutorial - https://youtu.be/hlv79rcHws0?si=WouU8XbXytVISCQe
# some changes were made in order to utilise a continuous action space

"""
Much of the code in this file may resemble code that was sumbitted for the CM30359 Coursework.
I adapted some code used for our TD3 implementation where possible and changed the rest where needed.
"""

import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn
import warnings

# Convert all UserWarnings into exceptions
warnings.filterwarnings("error", category=UserWarning)


from utils import pathJoin
from Memory import Memory
from torch.nn import (
    Linear,
    Softmax,
    LSTM,
    Sequential,
    init,
    Tanh,
    Parameter,
    functional as F,
)
from torch.distributions import Dirichlet
from torch.optim import Adam

device = torch.device("cpu")
_SAVE_SUFFIX = "_ppo"
_OPTIMISER_SAVE_SUFFIX = "_optimiser_ppo"


def layerInit(layer, std=np.sqrt(2), biasConst=0.0):
    """
    Function taken from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """
    init.orthogonal_(layer.weight, std)
    init.constant_(layer.bias, biasConst)
    return layer


class CriticNetwork(nn.Module):
    """
    Creates a critic network class for PPO
    input: state, hidden state, cell state
    output: valuation, hidden state, cell state
    """

    def __init__(
        self,
        fc1_n: int,
        fc2_n: int,
        lstmHiddenSize: int,
        state_n: int,
        actions_n: int,
        modelName: str,
    ):
        super(CriticNetwork, self).__init__()
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.lstmHiddenSize = lstmHiddenSize
        self.actions_n = actions_n
        self.state_n = state_n
        self.modelName = modelName

        self.save_file_name = self.modelName + _SAVE_SUFFIX
        self.optimiser_save_file_name = self.modelName + _OPTIMISER_SAVE_SUFFIX
        self.lstm = LSTM(
            input_size=self.state_n,
            hidden_size=lstmHiddenSize,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = layerInit(Linear(self.lstmHiddenSize, self.fc1_n))
        self.fc2 = layerInit(Linear(self.fc1_n, self.fc2_n))
        self.fc3 = layerInit(Linear(self.fc2_n, 1))
        self.tanh = nn.Tanh()

    def forward(self, state, hiddenState=None):
        if hiddenState is None:  # need to reset hidden states
            hiddenState = self.initHidden(batchSize=state.size(0))

        if state.dim() == 2:
            state = state.unsqueeze(1)
        lstmOut, (hidden, cell) = self.lstm(state, hiddenState)
        valuation = self.tanh(self.fc1(hidden[-1]))
        valuation = self.tanh(self.fc2(valuation))
        valuation = self.fc3(valuation)
        return valuation, (hidden, cell)

    def initHidden(self, batchSize=1):
        # reset hidden states
        h0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        c0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        return (h0, c0)


class ActorNetwork(nn.Module):
    """
    Creates an actor network class for PPO
    Input: state, hidden state, cell state
    Output: action, hidden state, cell state
    """

    def __init__(
        self,
        fc1_n: int,
        fc2_n: int,
        lstmHiddenSize: int,
        state_n: int,
        actions_n: int,
        modelName: str,
    ):
        super(ActorNetwork, self).__init__()
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.lstmHiddenSize = lstmHiddenSize
        self.state_n = state_n
        self.actions_n = actions_n
        self.modelName = modelName
        self.save_file_name = self.modelName + _SAVE_SUFFIX
        self.optimiser_save_file_name = self.modelName + _OPTIMISER_SAVE_SUFFIX

        # Map state to action
        self.lstm = LSTM(
            input_size=self.state_n,
            hidden_size=self.lstmHiddenSize,
            num_layers=1,
            batch_first=True,
        )

        self.fc1 = layerInit(Linear(self.lstmHiddenSize, self.fc1_n))
        self.fc2 = layerInit(Linear(self.fc1_n, self.fc2_n))
        self.alphaLayer = layerInit(Linear(self.fc2_n, actions_n), std=0.05)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def initHidden(self, batchSize=1):
        # reset hidden states
        h0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        c0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        return (h0, c0)

    def forward(self, state, hiddenState=None):
        """
        Computes the action distribution given the current state and hidden state.
        A small constant (1e-3) is added after softplus to ensure Dirichlet parameters
        remain strictly positive and avoid instability.
        Evidence for this can be found on this page: https://github.com/ray-project/ray/blob/master/rllib/models/torch/torch_action_dist.py
        """
        # reset hidden states
        if hiddenState is None:
            hiddenState = self.initHidden(batchSize=state.size(0))

        # To make input 3 dimensional.
        if state.dim() == 2:
            state = state.unsqueeze(1)

        lstmOut, (hidden, cell) = self.lstm(state, hiddenState)
        x = self.relu(self.fc1(hidden[-1]))
        x = self.relu(self.fc2(x))
        alpha = F.softplus(self.alphaLayer(x)) + 1e-3
        dist = torch.distributions.Dirichlet(alpha)
        return dist, (hidden, cell)


class PPOAgent:
    """
    Creates an agent class for PPO

    """

    def __init__(
        self,
        alpha: float,  # actor learnign rate
        policyClip: float,
        gamma: float,  # discount factor
        actor_noise: float,  # noise for actor
        lstmHiddenSize: int,
        state_n: int,
        actions_n: int,
        batch_size: int,
        fc1_n: int,  # number of neurons in first hidden layer
        fc2_n: int,  # number of neurons in second hidden layer
        gaeLambda: int = 0.95,
        epochs=10,
        riskAversion=0,
        featureExtractor=None,
        nonFeatureStateDim=None,
        maxSize=None,
    ):
        self.alpha = alpha
        self.policyClip = policyClip
        self.gamma = gamma
        self.noise = actor_noise
        self.state_n = state_n
        self.lstmHiddenSize = lstmHiddenSize
        self.actions_n = actions_n
        self.batch_size = batch_size
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.gaeLambda = gaeLambda
        self.epochs = epochs
        self.nonFeatureStateDim = nonFeatureStateDim
        self.memory = Memory(
            maxSize=maxSize,
            batchSize=batch_size,
            stateDim=nonFeatureStateDim,
            actionDim=actions_n,
            device=device,
            hiddenAndCellSize=lstmHiddenSize,
        )
        self.learn_step_count = 0
        self.time_step = 0
        self.riskAversion = riskAversion
        self.featureExtractor = featureExtractor.to(device)

        self.actor = ActorNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSize,
            self.state_n,
            self.actions_n,
            modelName="actor",
        ).to(device)

        self.critic = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSize,
            self.state_n,
            self.actions_n,
            modelName="critic",
        ).to(device)
        """
        # Optimizer. Since I backpropagate the feature extractor based on the actor and critic,
        # I need to include it in the optimizer as well.
        # Earlier in implementation, I wasn't certain how to backpropagate the sum 
        # actor and critic loss through the feature extractor, so I just decided to merge
        # the three optimizers into one.
        """
        self.optimizer = torch.optim.Adam(
            (
                list(self.actor.parameters())
                + list(self.critic.parameters())
                + list(featureExtractor.parameters())
            ),
            lr=self.alpha,
        )

    def select_action(
        self, observation, hiddenAndCellStates, sampling=True, returnHidden=False
    ):
        with torch.no_grad():
            """
            Action sampling and selection. The action is sampled from the Dirichlet distribution
            and the critic valuation is calculated using the critic network.
            However, during evaluation and testing, the action is selected as the mean of the distribution.
            """
            state = observation
            distribution, actorHidden = self.actor(state, hiddenAndCellStates["actor"])
            if sampling:
                action = distribution.sample()
            else:
                action = distribution.mean
            criticValuation, criticHidden = self.critic(
                state, hiddenAndCellStates["critic"]
            )
            probabilities = distribution.log_prob(action)  # assumes independence
            action = torch.squeeze(action)
        # self.time_step += 1 # not actually necessary to care about luckily
        if returnHidden:
            return action, probabilities, criticValuation, actorHidden, criticHidden
        else:
            return action, probabilities, criticValuation

    def store(
        self, state, action, probabilities, valuations, reward, done, hiddenStates
    ):
        self.memory.store(
            state, action, probabilities, valuations, reward, done, hiddenStates
        )

    def train(self, nextObs, criticHandC):
        (
            stateArr,
            actionArr,
            oldProbArr,
            valsArr,
            rewardArr,
            donesArr,
            actorH,
            actorC,
            criticH,
            criticC,
            featureH,  # unused
            featureC,  # unused
            batches,
        ) = self.memory.generateBatches()

        # append 0 to the end of the valuation array if terminal state else next state valuation
        # bootstrapping - need to generate valuation for the last state
        if not donesArr[-1] and nextObs is not None:  # if not terminal at the end
            with torch.no_grad():
                finalFeatures, _ = self.featureExtractor(nextObs)
                finalFeatures = finalFeatures.unsqueeze(1)  # batch dimension
                finalValuation, _ = self.critic(finalFeatures, criticHandC)
                finalValuation = finalValuation.squeeze(0).detach()
        else:  # - equivalently, nextObs is none if this is the case, since the next state simply cannot be computed
            finalValuation = torch.tensor([0.0], device=device)

        # add it to the end
        values = valsArr.squeeze().detach()
        try:
            values = torch.cat([values, finalValuation])
        except RuntimeError:
            raise ValueError("Values and finalValuation could not be concatenated.")

        advantage = torch.zeros(len(rewardArr), dtype=torch.float32, device=device)

        # TRUNCATED Generalized advantage estimation (GAE) calculation
        # delta = td_error
        # advantage estimation inspired by that found on: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        lastgaelam = 0

        for t in reversed(range(len(rewardArr))):
            nextnonterminal = (
                1 - donesArr[t].item()
            )  # Whether next state is non-terminal - accounts for episode distinctions
            nextvalue = values[t + 1]

            delta = rewardArr[t] + self.gamma * nextvalue * nextnonterminal - values[t]
            advantage[t] = lastgaelam = (
                delta + self.gamma * self.gaeLambda * nextnonterminal * lastgaelam
            )

        for batch in batches:
            actorHidden = (
                actorH[batch].unsqueeze(0),
                actorC[batch].unsqueeze(0),
            )
            criticHidden = (
                criticH[batch].unsqueeze(0),
                criticC[batch].unsqueeze(0),
            )
            states = stateArr[batch]
            actions = actionArr[batch]
            oldProbs = oldProbArr[batch].squeeze()
            adv = advantage[batch].detach()
            oldVals = values[batch]

            # Recompute featuers to ensure that actor and critic are always processing off
            # up-to-date representations
            features, _ = self.featureExtractor(states)
            features = features.unsqueeze(1)  # crtical to be processed as batch

            """
            Much of this follows the same logic as Phil Tabor's PPO implementation:
            """
            actorDist, _ = self.actor(features, actorHidden)
            criticOut, _ = self.critic(features, criticHidden)

            criticOut = criticOut.squeeze(
                -1
            )  # now required for possible 1 - length batch

            newProbs = actorDist.log_prob(
                actions
            )  # results in 0 distribution shift for batchsize=maxsize, but that means it works

            probRatio = torch.exp(newProbs - oldProbs)
            weightedProbs = adv * probRatio
            clippedWeightedProbs = adv * torch.clamp(
                probRatio, 1 - self.policyClip, 1 + self.policyClip
            )
            actorLoss = -torch.min(weightedProbs, clippedWeightedProbs).mean()

            returns = adv + oldVals
            try:
                criticLoss = F.mse_loss(criticOut, returns)
            except UserWarning as e:
                raise RuntimeError(f"Shape mismatch warning encountered: {e}")

            totalLoss = actorLoss + 0.5 * criticLoss

            self.optimizer.zero_grad()
            totalLoss.backward()
            # Gradient Clipping - same as Zou et al. (2024)
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]["params"], max_norm=0.5
            )
            self.optimizer.step()

        self.memory.clear()

    def save(self, save_dir: str):
        """
        These save functions are copied from our CM30359 Coursework utlizing TD3.
        """

        torch.save(
            self.critic.state_dict(), pathJoin(save_dir, self.critic.save_file_name)
        )
        torch.save(
            self.optimizer.state_dict(),
            pathJoin(save_dir, self.critic.optimiser_save_file_name),
        )

        torch.save(
            self.actor.state_dict(), pathJoin(save_dir, self.actor.save_file_name)
        )

        torch.save(
            self.featureExtractor.state_dict(),
            pathJoin(save_dir, self.featureExtractor.save_file_name),
        )

    def load(self, save_dir: str):
        self.critic.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.critic.save_file_name),
                weights_only=True,
            )
        )
        self.optimizer.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.critic.optimiser_save_file_name),
                weights_only=True,
            )
        )

        self.actor.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.actor.save_file_name),
                weights_only=True,
            )
        )
        self.featureExtractor.load_state_dict(
            torch.load(
                pathJoin(save_dir, self.featureExtractor.save_file_name),
                weights_only=True,
            )
        )
