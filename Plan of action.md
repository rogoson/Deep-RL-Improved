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
  1. WeightsAndBiases Setup - Good reason to think the issue was the normalisation. It literally turns all the data into noise and makes the problem virtually unsolvable. - _Partly Done, needs doing for other (non-noise) experiments_

## Second Week

- **Day 1:**
  1. Check all work done so far. Write extra verification tests. Add flags (to make stuff readable) - _Done_
  2. Ideally be able to visualise everything if possible. - _Done_
  3. Figure out why there is variation between runs even with seeds. Something is wrong (maybe try offline)? - wasn't seeding initialisation. - _Done_
- **Day 2:** Restructure experimentation. Noise -> Hyper -> Testing doesn't really make sense.
  1. Watch - _Done_
  2. Rename layers for wandb - _Done_
  3. Restore GPU usage - _Done_
  4. Test PPOLSTM
  5. It may be better to get rid of the hyperparameter sweep altogether. Just go straight for it?
  6. Test Log-Scaled, DSR, Log, CVaR - OFFLINE MODE IS FASTER PROBABLY
  - Serious vanishing Gradient problems for unscaled reward. Take a look at gradients on noise run - nonexistent.
  7. Learning Curves - you'll need to plot learning curves
- **Day 3:** Redo Introduction, Literature Review
- **Day 4-5:** Write/Rewrite Methodology section
- **Day 6-7:** _Buffer_ (catch up on any work, verify algorithm (_recurrent PPO_) methodology)

## Third Week

- **Day 1:** Review Bug File (maybe too harsh)
- **Day 2-3:** Rewrite experimental design
- **Day 4-5:** Run noise testing, rest of experiments
- **Day 6-7:** Write results and conclusions

## Fourth Week

- **Day 1:** Read through report - _introduction, literature review_
- **Day 2:** Read through report - _methodology, Experimental design_
- **Day 3:** Read through report - _experiments and conclusions_
- **Day 4:** Read through report - _conclusions_
- **Day 5-7:** _Buffer period_
