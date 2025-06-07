# Plan of Action

## First Week

- **Day 1:** Get new financial data and clean up indicators - _Done_
- **Day 2:** Clean up data and normalisation method - _Done_
- **Day 3:** Clean up formulas (_reward functions_). - _Done_
- **Day 4:** Re-scale log reward, comment DSR. - _Done_
- **Day 5:** Clean up algorithm training.
  1. GAE - _Done_
  2. Bootstrapping - _Done_
  3. Hidden states - _Done_
  4. Log prob - Entropy loss may help (distribution is peaked). - _Done_
- **Day 6:**
  - Test Entropy - _Done_
  - Test Advantage normalisation for small rewards. Neither worked, the entropy converges - _Done_
- **Day 7:** Implement Tests/Logs for _Everything_.
  1. WeightsAndBiases Setup - Good reason to think the issue was the normalisation (nope - lucky run used normalisation). It literally turns all the data into noise and makes the problem virtually unsolvable. - _Partly Done, needs doing for other (non-noise) experiments_

## Second Week

- **Day 1:**
  1. Check all work done so far. Write extra verification tests. Add flags (to make stuff readable) - _Done_
  2. Ideally be able to visualise everything if possible. - _Done_
  3. Figure out why there is variation between runs even with seeds. Something is wrong (maybe try offline)? - wasn't seeding initialisation. - _Done_
- **Day 2:** Restructure experimentation. Noise -> Hyper -> Testing doesn't really make sense.
  1. Watch - _Done_
  2. Rename layers for wandb - _Done_
  3. Restore GPU usage - _Done_
  4. Test PPOLSTM - _Failed_. No luck on a masked/unmasked pendulum with the full arch.
- **Day 3:** Re-plan. Figure out how to package stuff - _Done_
- **Day 4:** Restructure experimentation. Noise -> Hyper -> Testing doesn't really make sense.
  1. It may be better to get rid of the hyperparameter sweep altogether. Yes. Vary reward hyperparameters - _Done_
  2. Test Log-Scaled, DSR, Unscaled Log, CVaR - OFFLINE MODE IS FASTER PROBABLY
  3. Learning Curves - you'll need to plot learning curves - _clearly_
  4. Maybe try to fit a line for total reward compared to evaluation performances. - _no, time constraints_
  5. Set hidden size to 512 for ensuing experiments. - _Done_
- **Day 5:** Set up experiments
  1. Validate that everything is set up correctly. - _Done_
  2. Ensure that you know what you want for each section. - _Done_
  3. Add visualise data code (training, validation, training+validation, test) - _Done_
  4. Figure out Tesla issue - _Done_ (required redownloading data)
  5. Stocks issue - _Done_
- **Day 6:**: Set up your experiments to run. Start running them.
  1. Normalisation - _Done_
  2. Noise
  3. Rewards
  - Variance
- **Day 7:** Review Bugs file (categorize into severity) - include:
  1. What it is
  2. Impact
  3. Whether it is fixed
  4. What I learned?
     Second - Review Marked-up diss

## Third Week

- **Day 1-2:** Redo Introduction, Literature Review
- **Day 3-4:** Write/Rewrite Methodology section
- **Day 5-6:** Rewrite experimental design
- **Day 7:** Write results and conclusions

## Fourth Week

- **Day 1-3:** Read through report
- **Day 4-7:**
  1. Practice viva questions
  2. Check how to diff code (new and old) using marketplace "Diff" thing

# Suggested Experimentation (Week 2, Day 5):

# Goal

- Investigate reward functions and mitigative abilities
- Correlation between CVaR risk aversion and risk mitigation
- Effect of noise on small dataset

# Experimental Design

## Introduction

- Remove hyperparameter details. Focus on CVaR variability, noise, and reward functions.

## Methodology

### Noise and Seeds

- Preliminary explanation
- Show effects of noise used on prices

### Normalisation

- Show normalised and unnormalised validation data (in the paper, not appendix) - vary seeds (no noise)
- Show total rewards also with the normalised data – it seems very noisy, unstable progress:  
  [Unstable Progress](https://wandb.ai/richardpogoson-none/RL-Portfolio-Management/runs/ihukwg4t?nw=nwuserrichardpogoson)

# Results

## 1. Noise – Effect on SLR

- Vary seeds for **200k timesteps** and provide commentary using graphs. Compare Training Rewards with validation rewards for each base seed used = 20 graphs. Long, but doable.

## 2. Varied Base Seeds on Rewards

- CVaR's mitigation ability – analyze all cases
- reward variance on Test-Set
- Same as before - compare general metrics

## 3. Comparisons with Indices

- Show what the agent is really doing (Appendix) – strange behavior
