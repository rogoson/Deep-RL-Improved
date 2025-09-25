import torch

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
USE_PRIORITY = True


class Memory:
    """
    Creates a memory buffer for storing experiences.
    """

    def __init__(
        self,
        maxSize,
        batchSize,
        stateDim,
        actionDim,
        hiddenAndCellSizeDictionary,
        cnnFeature=False,
        isTD3Buffer=False,
        device="cpu",
    ):
        self.maxSize = maxSize
        self.batchSize = batchSize
        self.ptr = 0
        self.device = device

        # Experience buffers
        self.states = torch.zeros(
            (maxSize, *stateDim), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(
            (maxSize, actionDim), dtype=torch.float32, device=device
        )
        self.probabilities = torch.zeros(
            (maxSize, 1), dtype=torch.float32, device=device
        )
        self.nextStates = torch.zeros(
            (maxSize, *stateDim), dtype=torch.float32, device=device
        )
        self.criticValues = torch.zeros(
            (maxSize, 1), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros((maxSize,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((maxSize,), dtype=torch.bool, device=device)
        self.isTD3Buffer = isTD3Buffer
        self.memoryFull = False
        self.cnnFeature = cnnFeature

        def makeBuffer(role):
            return {
                "h": torch.zeros(
                    (maxSize, hiddenAndCellSizeDictionary[role]),
                    dtype=torch.float32,
                    device=device,
                ),
                "c": torch.zeros(
                    (maxSize, hiddenAndCellSizeDictionary[role]),
                    dtype=torch.float32,
                    device=device,
                ),
            }

        self.hiddenStateBuffers = {
            "actor": makeBuffer("actor"),
            "critic": makeBuffer("critic"),
        }
        if not cnnFeature:
            self.hiddenStateBuffers["feature"] = makeBuffer("feature")

        if self.isTD3Buffer:
            self.hiddenStateBuffers["critic2"] = makeBuffer("critic2")
            if not cnnFeature:
                self.hiddenStateBuffers["targetFeature"] = makeBuffer("feature")
            self.hiddenStateBuffers["targetActor"] = makeBuffer("targetActor")
            self.hiddenStateBuffers["targetCritic"] = makeBuffer("targetCritic")
            self.hiddenStateBuffers["targetCritic2"] = makeBuffer("targetCritic2")

    def store(
        self,
        state=None,
        action=None,
        probability=None,
        nextState=None,
        value=None,
        reward=None,
        done=None,
        hiddenStates=None,
    ):
        """Store experience with all associated hidden states"""
        index = self.ptr

        # Store basic experience data
        self.states[index] = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        )
        self.actions[index] = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        )
        if not self.isTD3Buffer:
            self.probabilities[index] = torch.as_tensor(
                probability, dtype=torch.float32, device=self.device
            )
            self.criticValues[index] = torch.as_tensor(
                value, dtype=torch.float32, device=self.device
            )
        else:
            self.nextStates[index] = torch.as_tensor(
                nextState, dtype=torch.float32, device=self.device
            )
        self.rewards[index] = torch.tensor(
            reward, dtype=torch.float32, device=self.device
        )
        self.dones[index] = torch.tensor(done, dtype=torch.bool, device=self.device)
        self._cachedDist = None
        self._cachedPtr = None
        self.components = [
            c
            for c in (
                ["actor", "critic", "feature"]
                if not self.isTD3Buffer
                else [
                    "actor",
                    "critic",
                    "critic2",
                    "feature",
                    "targetActor",
                    "targetCritic",
                    "targetCritic2",
                    "targetFeature",
                ]
            )
            if not self.cnnFeature or c not in ("feature", "targetFeature")
        ]

        # Store all hidden states in organized manner
        for component in self.components:
            h, c = hiddenStates[component]
            self.hiddenStateBuffers[component]["h"][index] = h.detach().squeeze()
            self.hiddenStateBuffers[component]["c"][index] = c.detach().squeeze()

        self.ptr += 1
        self.memoryFull = self.ptr == self.maxSize
        if self.memoryFull:
            self.clear()

    def genDist(self):
        if (
            hasattr(self, "_cachedDist")
            and self._cachedPtr == self.ptr
            and self._cachedFull == self.memoryFull
        ):
            return self._cachedDist
        # regenerate if needed
        if self.memoryFull:
            amount = torch.arange(self.maxSize, device=self.device)
            pDist = (
                torch.cat(
                    (
                        amount[self.maxSize - self.ptr :],
                        amount[: self.maxSize - self.ptr],
                    )
                )
                + 1
            ) / (0.5 * self.maxSize * (self.maxSize + 1))
        else:
            pDist = (torch.arange(self.ptr, device=self.device) + 1) / (
                0.5 * self.ptr * (self.ptr + 1)
            )
        self._cachedDist = pDist
        self._cachedPtr = self.ptr
        self._cachedFull = self.memoryFull
        return pDist

    def sample(self):
        numSamples = self.ptr
        if not self.isTD3Buffer:
            indices = torch.randperm(numSamples, device=self.device)
            batchStarts = torch.arange(
                0, numSamples, self.batchSize, device=self.device
            )
            batches = [indices[i : i + self.batchSize] for i in batchStarts]
        else:
            if self.memoryFull:
                size = self.maxSize
            else:
                size = self.ptr
            indices = torch.multinomial(
                (
                    self.genDist()
                    if USE_PRIORITY
                    else torch.ones(size, device=self.device) / size
                ),
                self.batchSize,
                replacement=True,
            )

        select = slice(None, numSamples) if not self.isTD3Buffer else indices

        hiddenStates = {
            comp: {
                stateType: self.hiddenStateBuffers[comp][stateType][select]
                for stateType in ["h", "c"]
            }
            for comp in self.components
        }

        return (
            (
                self.states[:numSamples],
                self.actions[:numSamples],
                self.probabilities[:numSamples],
                self.criticValues[:numSamples],
                self.rewards[:numSamples],
                self.dones[:numSamples],
                hiddenStates,
                batches,
            )
            if not self.isTD3Buffer
            else (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.nextStates[indices],
                self.dones[indices],
                hiddenStates,
            )
        )

    def clear(self):
        self.ptr = 0
