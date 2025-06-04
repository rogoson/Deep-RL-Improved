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
- **Day 4-5:** Restructure experimentation. Noise -> Hyper -> Testing doesn't really make sense.
  1. Think about differential entropy. The max for k=25 is about -54.7. Lower alphas mean less lower (more negative) entropy, but higher alphas mean less variance. Adding entropy bonus makes values want to stay close to 1 (on average), but does that impede learning? Multiple signals, quite complicated.
  2. It may be better to get rid of the hyperparameter sweep altogether. Just go straight for it?
  3. Test Log-Scaled, DSR, Log, CVaR - OFFLINE MODE IS FASTER PROBABLY
  - Serious vanishing Gradient problems for unscaled reward. Take a look at gradients on noise run - nonexistent.
  4. Learning Curves - you'll need to plot learning curves
  5. Try without feature extractor?
- **Day 6:** Review Bugs file (categorize into severity) - include:
  1. What it is
  2. Impact
  3. Whether it is fixed
  4. What I learned?
     Second - Review Marked-up diss
- **Day 7:** Redo Introduction, Literature Review

## Third Week

- **Day 1:** Write/Rewrite Methodology section
- **Day 2-3:** Rewrite experimental design
- **Day 4-5:** Run experiments
- **Day 6-7:** Write results and conclusions

## Fourth Week

- **Day 1-3:** Read through report
- **Day 4-7:**
  1. Practice viva questions
  2. Check how to diff code (new and old) using marketplace "Diff" thing
