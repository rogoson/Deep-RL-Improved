import torch


device = torch.device("cpu")


class Memory:
    """
    Creates a memory buffer for storing experiences.
    This is inspired by the same code used by Phil Tabor in his PPO tutorial, but is tensorized
    for faster computation and also because numpy and pytorch don't like each other very much.
    """

    def __init__(self, maxSize, batchSize, stateDim, actionDim, device="cpu"):
        self.maxSize = maxSize
        self.batchSize = batchSize
        self.ptr = 0
        self.device = device

        self.states = torch.zeros(
            (maxSize, stateDim[0], stateDim[1]), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(
            (maxSize, actionDim), dtype=torch.float32, device=device
        )
        self.probabilities = torch.zeros(
            (maxSize, 1), dtype=torch.float32, device=device
        )
        self.criticValues = torch.zeros(
            (maxSize, 1), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros((maxSize,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((maxSize,), dtype=torch.bool, device=device)

    def store(self, state, action, probability, value, reward, done):
        self.states[self.ptr] = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        )
        self.actions[self.ptr] = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        )
        self.probabilities[self.ptr] = torch.as_tensor(
            probability, dtype=torch.float32, device=self.device
        )
        self.criticValues[self.ptr] = torch.as_tensor(
            value, dtype=torch.float32, device=self.device
        )
        self.rewards[self.ptr] = torch.tensor(
            reward, dtype=torch.float32, device=self.device
        )
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.bool, device=self.device)
        self.ptr += 1

    def generateBatches(self):
        numSamples = self.ptr
        indices = torch.randperm(numSamples, device=self.device)
        batchStarts = torch.arange(0, numSamples, self.batchSize, device=self.device)
        batches = [indices[i : i + self.batchSize] for i in batchStarts]

        return (
            self.states[:numSamples],
            self.actions[:numSamples],
            self.probabilities[:numSamples],
            self.criticValues[:numSamples],
            self.rewards[:numSamples],
            self.dones[:numSamples],
            batches,
        )

    def clear(self):
        self.ptr = 0
