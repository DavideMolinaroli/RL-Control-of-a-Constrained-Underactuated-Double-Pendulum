# RL Control of a Constrained Underactuated Double Pendulum
![Animaton](model_penalty_pi6/best_model_pi6.gif)

## Overview

This project investigates whether model-free reinforcement learning can enforce state constraints in an underactuated Pendubot system without relying on constrained optimization. It includes a custom Gymnasium environment and SAC training pipelines for both swing-up and stabilization, along with a constraint handling strategy based on logarithmic barrier penalties and curriculum learning. The resulting policy recovers from disturbances while respecting position and velocity constraints on state evolution.

This repository includes:
- Environment setup,  
- Training and evaluation scripts,  
- The final project report with figures and experimental results.

---

## Repository structure

```
├── controller/                # implementation of different controllers (given)
├── model/                     # mathematical model of the double pendulum (given)
├── trained_models/
│   ├── model_default/         # default model, i.e trained without penalties and disturbances
│   └── model_penalty_*/       # variants with constraint penalties
├── parameters/                # simulation parameters
├── simulation/                # environment definition
├── utils/                     # utilities such as reset and plotting functions
├── train.py                   # main SAC training script
├── train_loose.py             # relaxed constraint variant
├── train_strict.py            # strict constraint variant
├── evaluate.py                # policy evaluation script
├── paper.pdf                  # project paper/report
```

---

## Requirements

Recommended environment (Python ≥ 3.8):

- `numpy`
- `scipy`
- `matplotlib`
- `gymnasium`
- `torch`
- `stable-baselines3`

---

### Train a policy

```bash
# default configuration
python train.py

# variants
python train_loose.py
python train_strict.py
```

---

### Evaluate a trained policy

```bash
# default configuration
python train.py

# variants
python train_loose.py
python train_strict.py
```

For completeness, each model has its own training and evaluation script. 

Evaluations render the pendulum motion and outputs performance statistics.


