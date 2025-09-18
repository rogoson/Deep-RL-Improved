from main.agents.Memory import Memory
from torch.nn import (
    Linear,
    LSTM,
    init,
    Parameter,
    functional as F,
)
from torch.distributions import Dirichlet, Independent, Normal
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch
import torch.nn as nn
import warnings

# Convert all UserWarnings into exceptions
warnings.filterwarnings("error", category=UserWarning)


device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
_SAVE_SUFFIX = "_ppo"
_OPTIMISER_SAVE_SUFFIX = "_optimiser_ppo"
BASE_DIR = Path(__file__).parent


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
        self.criticLstm = LSTM(
            input_size=self.state_n,
            hidden_size=lstmHiddenSize,
            num_layers=1,
            batch_first=True,
            device=device,
        )
        self.criticFc1 = layerInit(
            Linear(self.lstmHiddenSize, self.fc1_n, device=device)
        )
        self.criticFc2 = layerInit(Linear(self.fc1_n, self.fc2_n, device=device))
        self.criticFc3 = layerInit(Linear(self.fc2_n, 1, device=device))
        self.tanh = nn.Tanh()

    def forward(self, state, hiddenState=None):
        if hiddenState is None:  # need to reset hidden states
            hiddenState = self.initHidden(batchSize=state.size(0))

        if state.dim() == 2:
            state = state.unsqueeze(1)
        self.criticLstm.flatten_parameters()
        lstmOut, (hidden, cell) = self.criticLstm(state, hiddenState)
        valuation = self.tanh(self.criticFc1(hidden[-1]))
        valuation = self.tanh(self.criticFc2(valuation))
        valuation = self.criticFc3(valuation)
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
        useDirichlet: bool = True,  # Whether to use Dirichlet or Mean/Std for actions
    ):
        super(ActorNetwork, self).__init__()
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.lstmHiddenSize = lstmHiddenSize
        self.state_n = state_n
        self.actions_n = actions_n
        self.modelName = modelName
        self.useDirichlet = useDirichlet
        self.save_file_name = self.modelName + _SAVE_SUFFIX
        self.optimiser_save_file_name = self.modelName + _OPTIMISER_SAVE_SUFFIX

        # Map state to action
        self.actorLstm = LSTM(
            input_size=self.state_n,
            hidden_size=self.lstmHiddenSize,
            num_layers=1,
            batch_first=True,
            device=device,
        )

        self.actorFc1 = layerInit(
            Linear(self.lstmHiddenSize, self.fc1_n, device=device)
        )
        self.actorFc2 = layerInit(Linear(self.fc1_n, self.fc2_n, device=device))

        # For dirichlet
        self.dirichletAlphaLayer = layerInit(
            Linear(self.fc2_n, actions_n, device=device), std=0.05
        )

        # For Gaussian (following iclr blog track)
        self.actorMeanLayer = layerInit(
            Linear(self.fc2_n, actions_n, device=device), std=0.05
        )
        self.actorLogStd = Parameter(
            torch.zeros(1, actions_n),
            requires_grad=True,
        )  # learnable log standard deviation

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
        self.actorLstm.flatten_parameters()
        lstmOut, (hidden, cell) = self.actorLstm(state, hiddenState)
        x = self.relu(self.actorFc1(hidden[-1]))
        x = self.relu(self.actorFc2(x))

        if self.useDirichlet:
            alpha = F.softplus(self.dirichletAlphaLayer(x)) + 1e-3
            dist = Dirichlet(alpha)
        else:
            mean = self.actorMeanLayer(x)
            log_std = self.actorLogStd.expand_as(mean)
            action_std = torch.exp(log_std)
            normDist = Normal(mean, action_std)
            dist = Independent(normDist, 1)  # Treat as joint
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
        actorNoise: float,  # noise for actor
        lstmHiddenSizeDictionary: dict,
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
        entropyCoefficient=0.01,
        rewardFunction=None,
        normAdvantages=False,
        useEntropy=False,
        useDirichlet=True,  # Whether to use Dirichlet distribution for actions (otherwise mean/std)
        log_concentration_heatmap=False,  # Whether to log concentration heatmap - depending on internet speeds, this can slow training severely (by like 3x)
        cnnFeature=False,
        experimentState=None,
    ):
        self.alpha = alpha
        self.policyClip = policyClip
        self.gamma = gamma
        self.noise = actorNoise
        self.state_n = state_n
        self.lstmHiddenSizeDictionary = lstmHiddenSizeDictionary
        self.actions_n = actions_n
        self.batch_size = batch_size
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.gaeLambda = gaeLambda
        self.epochs = epochs
        self.nonFeatureStateDim = nonFeatureStateDim
        self.hasCNNFeature = cnnFeature
        self.memory = Memory(
            maxSize=maxSize,
            batchSize=batch_size,
            stateDim=nonFeatureStateDim,
            actionDim=actions_n,
            device=device,
            hiddenAndCellSizeDictionary=lstmHiddenSizeDictionary,
            cnnFeature=cnnFeature,
            isTD3Buffer=False,
        )
        self.learn_step_count = 0
        self.time_step = 0
        self.riskAversion = riskAversion
        self.featureExtractor = featureExtractor.to(device)
        self.entropyCoefficient = entropyCoefficient
        self.rewardFunction = rewardFunction
        self.normAdvantages = normAdvantages
        self.useEntropy = useEntropy
        self.useDirichlet = useDirichlet
        self.log_concentration_heatmap = log_concentration_heatmap and useDirichlet
        self.experimentState = experimentState
        self.actor = ActorNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSizeDictionary["actor"],
            self.state_n,
            self.actions_n,
            modelName="actor",
            useDirichlet=useDirichlet,
        ).to(device)

        self.critic = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSizeDictionary["critic"],
            self.state_n,
            self.actions_n,
            modelName="critic",
        ).to(device)
        """
        # Optimizer - one.
        """
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.featureExtractor.parameters(),
                    "lr": self.alpha * 2.0,
                },  # push step more
                {
                    "params": list(self.actor.parameters())
                    + list(self.critic.parameters()),
                    "lr": self.alpha,
                },
            ]
        )

    def select_action(
        self, state, hiddenAndCellStates, sampling=True, returnHidden=False
    ):
        with torch.no_grad():
            """
            Action sampling and selection. The action is sampled from the Dirichlet distribution
            and the critic valuation is calculated using the critic network.
            However, during evaluation and testing, the action is greedily selected as the mean of the distribution.
            """
            distribution, actorHidden = self.actor(state, hiddenAndCellStates["actor"])
            if sampling:
                action = distribution.sample()
            else:
                action = distribution.mean
            criticValuation, criticHidden = self.critic(
                state, hiddenAndCellStates["critic"]
            )
            probabilities = distribution.log_prob(
                action
            )  # assumes independence when using normal distribution
            action = torch.squeeze(action)
        if returnHidden:
            return action, probabilities, criticValuation, actorHidden, criticHidden
        return action, probabilities, criticValuation

    def store(self, state, action, probability, valuation, reward, done, hiddenStates):
        self.memory.store(
            state=state,
            action=action,
            probability=probability,
            value=valuation,
            reward=reward,
            done=done,
            hiddenStates=hiddenStates,
        )

    def learn(self, nextObs, hAndC):
        for _ in range(self.epochs):
            (
                stateArr,
                actionArr,
                oldProbArr,
                valsArr,
                rewardArr,
                donesArr,
                hiddenStates,
                batches,
            ) = self.memory.sample()

            actorH = hiddenStates["actor"]["h"]
            actorC = hiddenStates["actor"]["c"]
            criticH = hiddenStates["critic"]["h"]
            criticC = hiddenStates["critic"]["c"]
            if not self.hasCNNFeature:
                featureH = hiddenStates["feature"]["h"]
                featureC = hiddenStates["feature"]["c"]

            # append 0 to the end of the valuation array if terminal state else next state valuation
            # bootstrapping - need to generate valuation for the last state
            if not donesArr[-1] and nextObs is not None:  # if not terminal at the end
                with torch.no_grad():
                    finalFeatures, _ = self.featureExtractor(
                        nextObs, hAndC["feature"] if not self.hasCNNFeature else None
                    )
                    finalFeatures = finalFeatures.unsqueeze(1)  # batch dimension
                    finalValuation, _ = self.critic(finalFeatures, hAndC["critic"])
                    finalValuation = finalValuation.squeeze(0).detach()
            else:  # - equivalently, nextObs is a vector of zeros, since the next state simply cannot be computed (and it doesn't matter)
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

                delta = (
                    rewardArr[t] + self.gamma * nextvalue * nextnonterminal - values[t]
                )
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
                if not self.hasCNNFeature:
                    featureHidden = (
                        featureH[batch].unsqueeze(0),
                        featureC[batch].unsqueeze(0),
                    )

                states = stateArr[batch]
                actions = actionArr[batch]
                oldProbs = oldProbArr[batch].squeeze()
                adv = advantage[batch].detach()
                oldVals = values[batch]

                # Recompute featuers to ensure that actor and critic are always processing off
                # up-to-date representations
                features, _ = self.featureExtractor(
                    states, featureHidden if not self.hasCNNFeature else None
                )
                features = features.unsqueeze(1)  # crtical to be processed as batch

                actorDist, _ = self.actor(features, actorHidden)
                criticOut, _ = self.critic(features, criticHidden)

                criticOut = criticOut.squeeze(-1)  # that was false

                newProbs = actorDist.log_prob(actions)

                if self.useEntropy:
                    # encourage exploration by adding entropy to the loss
                    entropy = actorDist.entropy().mean()
                else:
                    entropy = torch.tensor(0.0, device=device)

                # robbed from blog track also, should help with stability
                if self.normAdvantages:
                    # Normalize advantages to have mean 0 and std 1
                    normalisedAdv = (adv - adv.mean()) / (adv.std() + 1e-8)
                else:
                    normalisedAdv = adv

                probRatio = torch.exp(newProbs - oldProbs)
                weightedProbs = normalisedAdv * probRatio
                clippedWeightedProbs = normalisedAdv * torch.clamp(
                    probRatio, 1 - self.policyClip, 1 + self.policyClip
                )
                actorLoss = -torch.min(weightedProbs, clippedWeightedProbs).mean()

                # Use the original (unnormalized) advantages to compute returns
                returns = adv + oldVals
                try:
                    criticLoss = F.huber_loss(
                        criticOut, returns
                    )  # MSE gave MASSIVE losses (earlier on in implementation)
                except UserWarning as e:
                    raise RuntimeError(f"Shape mismatch warning encountered: {e}")

                totalLoss = (
                    actorLoss
                    + 0.5 * criticLoss
                    - (
                        self.entropyCoefficient * entropy / self.actions_n
                    )  # linearly scale down entropy contribution
                )

                self.optimizer.zero_grad()
                totalLoss.backward()
                # Gradient Clipping - same as Zou et al. (2024)
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]["params"], max_norm=1.0
                )
                self.optimizer.step()

        wandb.log(
            {
                "actor_loss": actorLoss.item(),
                "critic_loss": criticLoss.item(),
                "entropy": entropy.item() if self.useEntropy else 0.0,
                "total_loss": totalLoss.item(),
                "advantage_mean": normalisedAdv.mean().item(),
                "advantage_std": normalisedAdv.std().item(),
                "returns_mean": returns.mean().item(),
                "actor_prob_ratio_mean": probRatio.mean().item(),
            }
        )
        if self.log_concentration_heatmap:
            concentrations = actorDist.concentration.cpu().detach().numpy()

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                concentrations, cmap="viridis", annot=True, fmt=".2f", linewidths=0.5
            )
            plt.xlabel("Concentration Parameter Index")
            plt.ylabel("Distribution Index")
            plt.title("Heatmap of Concentration Parameters")
            plt.tight_layout()

            # Log the heatmap image to wandb
            wandb.log({"concentration_heatmap": wandb.Image(plt)})
            plt.close()
        self.memory.clear()

    def save(self, metric: float, index: str = ""):
        """
        Saves actor/critic/optimizer/featureExtractor only if `metric` beats previous best.
        AI supported save functions
        """
        sd = (
            Path(__file__).parent
            / index
            / self.__class__.__name__
            / Path(("CNNFeature" if self.hasCNNFeature else "LSTMFeature"))
            / self.experimentState
        )

        sd.mkdir(parents=True, exist_ok=True)

        # where we store best metric
        metric_file = sd / "best_return.txt"

        # read previous best (if exists)
        if metric_file.exists():
            try:
                prev_best = float(metric_file.read_text())
            except ValueError:
                prev_best = None
        else:
            prev_best = None

        # decide if this metric is an improvement
        if prev_best is None:
            improved = True
        else:
            improved = metric > prev_best

        if improved:
            # persist new best
            metric_file.write_text(f"{metric:.6f}")

            # now save all sub‐modules
            torch.save(self.critic.state_dict(), sd / self.critic.save_file_name)
            torch.save(
                self.optimizer.state_dict(), sd / self.critic.optimiser_save_file_name
            )
            torch.save(self.actor.state_dict(), sd / self.actor.save_file_name)
            torch.save(
                self.featureExtractor.state_dict(),
                sd / self.featureExtractor.save_file_name,
            )

            self.best_metric = metric
            print(f"New best return: {metric:.6f}. Checkpoints written to {sd}")
        else:
            print(
                f"— No improvement in return ({metric:.6f} vs {prev_best:.6f}); skipping save."
            )
        return improved, sd

    def load(self, save_dir: str, index: str = ""):
        sd = (
            Path(__file__).parent
            / index
            / self.__class__.__name__
            / Path(("CNNFeature" if self.hasCNNFeature else "LSTMFeature"))
            / self.experimentState
        )
        self.critic.load_state_dict(
            torch.load(sd / self.critic.save_file_name, weights_only=True)
        )
        self.optimizer.load_state_dict(
            torch.load(sd / self.critic.optimiser_save_file_name, weights_only=True)
        )
        self.actor.load_state_dict(
            torch.load(sd / self.actor.save_file_name, weights_only=True)
        )
        self.featureExtractor.load_state_dict(
            torch.load(sd / self.featureExtractor.save_file_name, weights_only=True)
        )
