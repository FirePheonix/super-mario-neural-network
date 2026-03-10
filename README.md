# Neuro Mario

A neuroevolution AI that learns to play Super Mario Bros using a genetic algorithm. No backpropagation — pure evolution: networks compete, the best survive, reproduce, and mutate.

## How It Works

- Each agent is a small 2-layer neural network (1024 → 128 → 4 outputs)
- A population of 60 agents plays Mario simultaneously
- Fitness = x-position reached + reward bonus - time penalty
- Top agents are kept (elitism), the rest are bred via tournament selection + crossover + mutation
- Mutation rate adapts automatically: shrinks on improvement, grows on stagnation

---

## Setup

### Prerequisites

- Python 3.11
- Windows (tested), Linux/macOS should work

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/neuro-mario.git
cd neuro-mario
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

### Step 3 — Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running

### Train the AI

Runs the evolutionary training loop. Opens up to 6 game windows in parallel (one per worker). Saves the best network to `models/best_network.pkl` automatically.

```bash
python train.py
```

To resume training from a previously saved model, just run the same command — it auto-loads `models/best_network.pkl` if it exists.

### Watch the AI play

Loads the best saved model and plays a single episode with a full-size game window.

```bash
python quick_play.py
```

If no trained model exists yet, it falls back to a random network so you can see the baseline.

---

## Configuration

Edit [config.py](config.py) to tune hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `POP_SIZE` | 60 | Number of agents per generation |
| `ELITE_SIZE` | 8 | Top agents kept unchanged each generation |
| `GENERATIONS` | 1000 | Total generations to train |
| `HIDDEN_SIZE` | 128 | Hidden layer size of each network |
| `INITIAL_MUTATION_RATE` | 0.02 | Starting mutation strength |
| `MAX_STEPS_PER_EPISODE` | 3000 | Max steps before episode ends |

## File Structure

```
neuro-mario/
├── config.py        # Hyperparameters
├── model.py         # Neural network (forward pass, clone, crossover helpers)
├── env_utils.py     # Mario environment setup, frame preprocessing, fitness evaluation
├── evolution.py     # Genetic algorithm (selection, crossover, mutation, adaptation)
├── train.py         # Main training loop with multiprocessing
├── quick_play.py    # Watch the trained agent play
├── requirements.txt
└── models/          # Saved model checkpoints (generated after training)
```
