"I robbed and adapted this from our RL Coursework in CM30359. A lot of things will look similar"

from main.agents.Memory import Memory
import copy
from torch.nn import Linear, LSTM, functional as F, ReLU, Tanh, Softmax
from pathlib import Path
from torch.optim import Adam
import torch
import torch.nn as nn
import warnings

# Convert all UserWarnings into exceptions
warnings.filterwarnings("error", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SAVE_SUFFIX = "_td3"
_OPTIMISER_SAVE_SUFFIX = "_optimiser_td3"


class CriticNetwork(nn.Module):
    """
    Creates a critic network class for TD3

    """

    def __init__(
        self,
        fc1_n: int,
        fc2_n: int,
        lstmHiddenSize: int,
        state_n: int,
        actions_n: int,
        model_name: str,
    ):
        super(CriticNetwork, self).__init__()
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.state_n = state_n
        self.actions_n = actions_n
        self.model_name = model_name
        self.lstmHiddenSize = lstmHiddenSize

        self.save_file_name = self.model_name + _SAVE_SUFFIX
        self.optimiser_save_file_name = self.model_name + _OPTIMISER_SAVE_SUFFIX

        # Take in concatenation of states and actions, output Q value
        input_n = self.state_n + self.actions_n
        self.criticLstm = LSTM(
            input_size=input_n,
            hidden_size=lstmHiddenSize,
            num_layers=1,
            batch_first=True,
            device=device,
        )
        self.criticFc1 = Linear(lstmHiddenSize, self.fc1_n, device=device)
        self.criticFc2 = Linear(self.fc1_n, self.fc2_n, device=device)
        self.criticFc3 = Linear(self.fc2_n, 1, device=device)
        self.tanh = Tanh()

    def initHidden(self, batchSize=1):
        # reset hidden states
        h0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        c0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        return (h0, c0)

    def forward(self, state: torch.Tensor, action: torch.Tensor, hiddenState=None):

        # reset hidden states
        if hiddenState is None:
            hiddenState = self.initHidden(batchSize=state.size(0))

        stateAction = torch.cat([state, action.to(device)], 1)
        # To make input 3 dimensional.
        if stateAction.dim() == 2:
            stateAction = stateAction.unsqueeze(
                1
            )  # need to check what happes shepwise here

        # Concat to fit input shape (state_n + actions_n)
        lstmOut, (hidden, cell) = self.criticLstm(stateAction, hiddenState)
        q = self.tanh(self.criticFc1(hidden[-1]))
        q = self.tanh(self.criticFc2(q))
        q = self.criticFc3(q)
        return q, (hidden, cell)  # return Q value and hidden states


class ActorNetwork(nn.Module):
    """
    Creates an actor network class for TD3

    """

    def __init__(
        self,
        fc1_n: int,
        fc2_n: int,
        lstmHiddenSize: int,
        state_n: int,
        actions_n: int,
        model_name: str,
    ):
        super(ActorNetwork, self).__init__()
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.state_n = state_n
        self.actions_n = actions_n
        self.model_name = model_name
        self.save_file_name = self.model_name + _SAVE_SUFFIX
        self.optimiser_save_file_name = self.model_name + _OPTIMISER_SAVE_SUFFIX
        self.lstmHiddenSize = lstmHiddenSize

        self.actorLstm = LSTM(
            input_size=self.state_n,
            hidden_size=lstmHiddenSize,
            num_layers=1,
            batch_first=True,
            device=device,
        )

        # Map state to action
        self.actorFc1 = Linear(lstmHiddenSize, self.fc1_n, device=device)
        self.actorFc2 = Linear(self.fc1_n, self.fc2_n, device=device)
        self.actorFc3 = Linear(self.fc2_n, actions_n, device=device)
        self.relu = ReLU()
        self.softMax = Softmax(dim=1)

    def initHidden(self, batchSize=1):
        # reset hidden states
        h0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        c0 = torch.zeros(1, batchSize, self.lstmHiddenSize).to(device)
        return (h0, c0)

    def forward(self, state, hiddenState=None):

        # reset hidden states
        if hiddenState is None:
            hiddenState = self.initHidden(batchSize=state.size(0))

        if state.dim() == 2:
            state = state.unsqueeze(1)  # need to check what happes shepwise here

        # Concat to fit input shape (state_n + actions_n)
        lstmOut, (hidden, cell) = self.actorLstm(state, hiddenState)

        state = self.relu(self.actorFc1(hidden[-1]))
        state = self.relu(self.actorFc2(state))
        state = self.softMax(self.actorFc3(state))

        return state, (hidden, cell)


class TD3Agent:
    """
    Creates an agent class for TD3

    """

    def __init__(
        self,
        alpha: float,  # actor learnign rate
        beta: float,  # critic learning rate
        gamma: float,  # discount factor
        tau: float,  # interpolation parameter
        actorNoise: float,  # noise for actor exploration
        targetNoise: float,  # noise for target networks
        state_n: int,
        actions_n: int,
        batchSize: int,
        fc1_n: int,  # number of neurons in first hidden layer
        fc2_n: int,  # number of neurons in second hidden layer
        actorUpdateFreq: int = 2,  # Update actor every n steps
        warmup: int = 500,
        maxSize: int = 50000,
        numberOfUpdates: int = 5,
        featureExtractor=None,
        nonFeatureStateDim=None,
        lstmHiddenSizeDictionary=None,
        rewardFunction=None,
        experimentState=None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.noise = actorNoise
        self.targetNoise = targetNoise
        self.state_n = state_n
        self.actions_n = actions_n
        self.batchSize = batchSize
        self.fc1_n = fc1_n
        self.fc2_n = fc2_n
        self.actorUpdateFreq = actorUpdateFreq
        self.warmup = warmup
        self.maxSize = maxSize
        self.numberOfUpdates = numberOfUpdates
        self.lstmHiddenSizeDictionary = lstmHiddenSizeDictionary
        self.experimentState = experimentState
        self.featureExtractor = featureExtractor
        self.targetFeatureExtractor = copy.deepcopy(featureExtractor)
        self.rewardFunction = rewardFunction
        self.memory = Memory(
            maxSize=maxSize,
            batchSize=batchSize,
            stateDim=nonFeatureStateDim,
            actionDim=actions_n,
            device=device,
            hiddenAndCellSizeDictionary=lstmHiddenSizeDictionary,
            isTD3Buffer=True,
        )
        self.learnStepCount = 0
        self.timeStep = 0

        self.actor = ActorNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSizeDictionary["actor"],
            self.state_n,
            self.actions_n,
            model_name="actor",
        ).to(device)
        self.actorOptimiser = Adam(
            list(self.actor.parameters()) + list(self.featureExtractor.parameters()),
            lr=self.alpha,
        )

        self.critic = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSizeDictionary["critic"],
            self.state_n,
            self.actions_n,
            model_name="critic",
        ).to(device)
        self.criticOptimiser = Adam(
            list(self.critic.parameters()) + list(self.featureExtractor.parameters()),
            lr=self.beta,
        )  # would probably be best to just have one optimiser - feature extractor might get stepped in both directions

        self.critic2 = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSizeDictionary["critic"],
            self.state_n,
            self.actions_n,
            model_name="critic2",
        ).to(device)
        self.critic2Optimiser = Adam(
            list(self.critic2.parameters()) + list(self.featureExtractor.parameters()),
            lr=self.beta,
        )

        self.targetActor = ActorNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSizeDictionary["actor"],
            self.state_n,
            self.actions_n,
            model_name="targetActor",
        ).to(device)

        self.targetCritic = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSizeDictionary["critic"],
            self.state_n,
            self.actions_n,
            model_name="targetCritic",
        ).to(device)

        self.targetCritic2 = CriticNetwork(
            self.fc1_n,
            self.fc2_n,
            self.lstmHiddenSizeDictionary["critic"],
            self.state_n,
            self.actions_n,
            model_name="targetCritic2",
        ).to(device)

        self.updateNetwork(self.critic, self.targetCritic)
        self.updateNetwork(self.critic2, self.targetCritic2)
        self.updateNetwork(self.actor, self.targetActor)
        self.updateNetwork(self.featureExtractor, self.targetFeatureExtractor)

    def select_action(
        self, observation, hiddenAndCellStates, returnHidden=False, useNoise=True
    ):

        state = observation
        with torch.no_grad():
            action, actorHidden = self.actor(state, hiddenAndCellStates["actor"])
            _, criticHidden = self.critic(state, action, hiddenAndCellStates["critic"])
            _, critic2Hidden = self.critic2(
                state, action, hiddenAndCellStates["critic2"]
            )
            _, targetActorHidden = self.targetActor(
                state, hiddenAndCellStates["targetActor"]
            )
            _, targetCriticHidden = self.targetCritic(
                state, action, hiddenAndCellStates["targetCritic"]
            )
            _, targetCritic2Hidden = self.targetCritic2(
                state, action, hiddenAndCellStates["targetCritic2"]
            )

            noise = torch.randn_like(action) * self.noise
            noise = noise.clamp(
                self.noise * -3, self.noise * 3
            )  # no more than 3std (99.7%)
            action = (action + noise).clamp(0, 1)
        self.timeStep += 1

        if returnHidden:
            return (
                action.clamp(0, 1).squeeze(),
                actorHidden,
                criticHidden,
                critic2Hidden,
                targetActorHidden,
                targetCriticHidden,
                targetCritic2Hidden,
            )
        return action.clamp(0, 1).squeeze()

    def updateNetwork(self, network, targetNetwork):
        """
        Perform soft update on networks
        """
        for param, targetParam in zip(network.parameters(), targetNetwork.parameters()):
            targetParam.data.copy_(
                self.tau * param.data + (1 - self.tau) * targetParam.data
            )

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        nextState: torch.Tensor,
        done: bool,
        hiddenStates: dict,
    ):
        self.memory.store(
            state=state,
            action=action,
            reward=reward,
            nextState=nextState,
            done=done,
            hiddenStates=hiddenStates,
        )

    def learn(self, nextObs=None, hAndC=None):  # unused
        # If there aren't enough experiences stored to fairly sample
        # self.numberOfUpdates updates, use as many as possible
        amountOfExperienceStored = self.memory.ptr // self.batchSize
        for update in range(min(amountOfExperienceStored, self.numberOfUpdates)):
            states, actions, rewards, newStates, doneArr, hiddenStates = (
                self.memory.sample()
            )

            components = [
                "actor",
                "critic",
                "critic2",
                "feature",
                "targetFeature",
                "targetActor",
                "targetCritic",
                "targetCritic2",
            ]

            hiddenDict = {}
            for name in components:
                h = hiddenStates[name]["h"].unsqueeze(0)
                c = hiddenStates[name]["c"].unsqueeze(0)
                hiddenDict[f"{name}Hidden"] = (h, c)

            states = states.to(device)
            featureVecs, _ = self.featureExtractor.forward(
                states, hiddenDict["featureHidden"]
            )
            actions = actions.to(device)
            rewards = rewards.to(device).unsqueeze(1)
            newStates = newStates.to(device)
            newTargetFeatureVecs, _ = self.targetFeatureExtractor.forward(
                newStates, hiddenDict["targetFeatureHidden"]
            )
            doneArr = doneArr.to(device).unsqueeze(1)  # unsqueeze more

            with torch.no_grad():
                targetActions, _ = self.targetActor(
                    newTargetFeatureVecs, hiddenDict["targetActorHidden"]
                )
                noise = torch.randn_like(targetActions) * self.targetNoise
                noise = noise.clamp(self.targetNoise * -3, self.targetNoise * 3)
                targetActions = (targetActions + noise).clamp(0, 1)

                q1, _ = self.targetCritic(
                    newTargetFeatureVecs,
                    targetActions,
                    hiddenDict["targetCriticHidden"],
                )
                q2, _ = self.targetCritic2(
                    newTargetFeatureVecs,
                    targetActions,
                    hiddenDict["targetCritic2Hidden"],
                )

                targetQValue = torch.min(
                    q1, q2
                )  # Take min to reduce overestimation bias
                targetQValue = rewards + ~doneArr * self.gamma * targetQValue

            q1, _ = self.critic(featureVecs, actions, hiddenDict["criticHidden"])
            q2, _ = self.critic2(featureVecs, actions, hiddenDict["critic2Hidden"])

            criticLoss = F.mse_loss(targetQValue, q1)
            critic2Loss = F.mse_loss(targetQValue, q2)
            totalCriticLoss = criticLoss + critic2Loss

            self.criticOptimiser.zero_grad()
            self.critic2Optimiser.zero_grad()
            totalCriticLoss.backward()
            self.criticOptimiser.step()
            self.critic2Optimiser.step()

            self.learnStepCount += 1
            # update actor every other time step
            if self.learnStepCount % self.actorUpdateFreq == 0:
                featureVecs_actor, _ = self.featureExtractor.forward(
                    states, hiddenDict["featureHidden"]
                )
                newActions, _ = self.actor(featureVecs_actor, hiddenDict["actorHidden"])
                criticValuation, _ = self.critic(
                    featureVecs_actor, newActions, hiddenDict["criticHidden"]
                )
                actorLoss = -criticValuation.mean()
                self.actorOptimiser.zero_grad()
                actorLoss.backward()
                self.actorOptimiser.step()

                self.updateNetwork(self.critic, self.targetCritic)
                self.updateNetwork(self.critic2, self.targetCritic2)
                self.updateNetwork(self.actor, self.targetActor)

    def save(self, metric: float, index: str = "") -> bool:
        """
        Saves all TD3 networks and optimizers only if `metric`
        (e.g. episode return) exceeds the previous best.
        """
        # build checkpoint directory
        sd = (
            Path(__file__).parent
            / index
            / self.__class__.__name__
            / self.experimentState
        )
        sd.mkdir(parents=True, exist_ok=True)

        # file to store best metric so far
        metricFile = sd / "best_return.txt"

        # read existing best (if any)
        if metricFile.exists():
            try:
                previousBest = float(metricFile.read_text())
            except ValueError:
                previousBest = None
        else:
            previousBest = None

        # check for improvement
        improved = (previousBest is None) or (metric > previousBest)

        if improved:
            # persist new best metric
            metricFile.write_text(f"{metric:.6f}")
            self.bestMetric = metric

            # save critics and their optimizers
            torch.save(self.critic.state_dict(), sd / self.critic.save_file_name)
            torch.save(
                self.criticOptimiser.state_dict(),
                sd / self.critic.optimiser_save_file_name,
            )

            torch.save(
                self.targetCritic.state_dict(),
                sd / self.targetCritic.save_file_name,
            )

            torch.save(self.critic2.state_dict(), sd / self.critic2.save_file_name)
            torch.save(
                self.critic2Optimiser.state_dict(),
                sd / self.critic2.optimiser_save_file_name,
            )

            torch.save(
                self.targetCritic2.state_dict(),
                sd / self.targetCritic2.save_file_name,
            )

            # save actors and their optimizers
            torch.save(self.actor.state_dict(), sd / self.actor.save_file_name)
            torch.save(
                self.actorOptimiser.state_dict(),
                sd / self.actor.optimiser_save_file_name,
            )

            torch.save(
                self.targetActor.state_dict(), sd / self.targetActor.save_file_name
            )

            print(f"New best return: {metric:.6f}. Checkpoints written to: {sd}")
        else:
            print(
                f"— No improvement ({metric:.6f} vs {previousBest:.6f}); skipping save."
            )

        return improved

    def load(self, save_dir: str = None, index: str = ""):
        """
        Loads the best‐so‐far checkpoints for all TD3 networks
        and optimizers from the experiment directory.
        """
        sd = Path(__file__).parent / self.__class__.__name__ / self.experimentState

        # critics
        self.critic.load_state_dict(
            torch.load(sd / self.critic.save_file_name, weights_only=True)
        )
        self.criticOptimiser.load_state_dict(
            torch.load(sd / self.critic.optimiser_save_file_name, weights_only=True)
        )

        self.targetCritic.load_state_dict(
            torch.load(sd / self.targetCritic.save_file_name, weights_only=True)
        )

        self.critic2.load_state_dict(
            torch.load(sd / self.critic2.save_file_name, weights_only=True)
        )
        self.critic2Optimiser.load_state_dict(
            torch.load(sd / self.critic2.optimiser_save_file_name, weights_only=True)
        )

        self.targetCritic2.load_state_dict(
            torch.load(sd / self.targetCritic2.save_file_name, weights_only=True)
        )

        # actors
        self.actor.load_state_dict(
            torch.load(sd / self.actor.save_file_name, weights_only=True)
        )
        self.actorOptimiser.load_state_dict(
            torch.load(sd / self.actor.optimiser_save_file_name, weights_only=True)
        )

        self.targetActor.load_state_dict(
            torch.load(sd / self.targetActor.save_file_name, weights_only=True)
        )

        print(f"Loaded checkpoints from: {sd}")
