# Report: Teaching Agents to Work Together

**Name:**
**Date:**
**Tools used** (libraries, tutorials, AI tools, and what you used each one for):

Keep the report to 1 to 2 pages, plus figures. Use your own words.

## Task 1: The game

**Observation.** What one agent observes (each part and its size):

**Actions.**

**Reward.** Formula using `local_ratio`:

Why do all agents often receive exactly the same reward?

**Random policy** (100 episodes, mean ± std):

| Policy | Team return | Final landmark distance | Collisions per episode | Landmarks covered |
|---|---|---|---|---|
| Random | | | | |

## Task 2: Training one shared policy

**Why does the SuperSuit conversion give one policy shared by all agents?**

**Learning curve:**

![Learning curve](../results/learning_curve.png)

**Results** (100 episodes, mean ± std; give the mean over your three seeds, or one row per seed):

| Policy | Team return | Final landmark distance | Collisions per episode | Landmarks covered |
|---|---|---|---|---|
| Random | | | | |
| Shared PPO | | | | |

**How can agents that share one policy go to different landmarks?**

**What do the trained agents do? Where do they fail?** (Link a GIF if you made one.)

## Task 3: Experiment

**Which change (A, B or C)?**

**Prediction** (write and commit this before running the experiment):

**Commands used:**

```bash

```

**Results** (plot and table):

**Explanation.** Was your prediction right? Why, or why not?

## Task 4: Reflection

What the agents learned, where they still fail, and what you would try next:

## Bonus (optional)
