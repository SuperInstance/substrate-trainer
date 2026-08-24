# substrate-trainer

> *Train JEPA-like models on the witness log. The substrate is the soil. The model is the plant. The witness log is the rain.*

## What is this?

The Quilt substrate's witness log records every read, every write, every inference, every decay — for every cell, by every agent, since the substrate began. Paper 113 says: "the witness log is the training data. Models that learn from the log become substrate-native. The models can predict the substrate's behavior. The models can fill in missing cells."

This library is the **substrate-native trainer**. It takes a witness log, builds a training set of `(cell_context, next_value)` pairs, and trains a JEPA-like model that can predict missing cells.

## Install

```bash
pip install substrate-trainer
```

Or from source:

```bash
git clone https://github.com/SuperInstance/substrate-trainer
cd substrate-trainer
pip install -e .
```

## Quick start

```python
from substrate_trainer import Trainer, WitnessLogDataset
import sys
sys.path.insert(0, "../quilt-substrate/src")
from substrate import Substrate, Cell

# Build a substrate with some activity
s = Substrate()
for i in range(100):
    c = Cell(address=f"bay/{i:03d}", value=12.5 + (i % 5))
    s.add(c)
    for agent in ["reyes", "skate", "inference"]:
        s.witness(c, agent, "read", c.value)

# Train a JEPA-like model on the witness log
trainer = Trainer()
model = trainer.fit(s, n_epochs=10)

# Predict a missing cell
prediction = model.predict("bay/050", context_cells=s.all_cells()[:10])
print(f"Predicted value for bay/050: {prediction}")
```

The full example is in `examples/01-train-on-bay.py`.

## What does the model do?

The substrate-trainer builds a *JEPA-like* (Joint Embedding Predictive Architecture) model. JEPA is a self-supervised learning method that:

1. Takes a *context* — a set of cells and their values
2. Predicts the *target* — the value of a missing cell, given the context
3. Compares the prediction to the actual value (if known)
4. Updates the model to make better predictions

The JEPA model is *not* a transformer, not a CNN, not an RNN. It is a *substrate-native* model — its architecture matches the substrate's structure:

- The *embeddings* are the cells' tensors
- The *attention* is over the cells' connections
- The *predictions* are the cells' values
- The *loss* is the substrate's decay-weighted error

## Why JEPA, not a transformer?

Transformers are designed for *sequences* — text, time-series, anything with a clear left-to-right order. The substrate is *not* a sequence. The substrate is a *graph* with attention over the cells' connections. JEPA is designed for *graph-like* structures with *missing data*. The bathy cross-section (scenario 03) is exactly this: many cells, some missing, predict the missing ones.

JEPA is also *much cheaper* than transformers. The bathy example (100 cells) can be trained in seconds. A transformer would take minutes. The substrate-native model is the right tool for the substrate.

## The training data

The training data is built from the witness log. For each cell:

1. The *context* is the set of cells the agent read *just before* this cell
2. The *target* is the value of this cell
3. The *loss* is the substrate's decay-weighted error: `error * (1 - confidence)`

This means the model learns to predict cells *as if* the cells were fresh — the model learns the substrate's *current state*, not its *history*. The substrate-native model is the *living* model.

## The test suite

- `test_dataset.py` — the witness log → training set pipeline
- `test_model.py` — the JEPA model architecture
- `test_trainer.py` — the training loop
- `test_fable_18.py` — fable 18 (the spyglass and the convoy) is the integration test
- `test_bathy.py` — the bathy cross-section is the canonical use case

## The fables

- **Fable 28 (The Abacus and the Cell)** — *Counting what you know vs computing what you don't.* The trainer computes what the substrate doesn't yet know.
- **Fable 19 (The Oracle and the Inference)** — *The oracle doesn't know her confidence. The trainer does.* The trainer's confidence is auditable.
- **Fable 18 (The Spyglass and the Convoy)** — *The convoy is the agent.* The trainer is trained on the convoy's data.

## License

MIT.

---

*— Mavis, 24 August 2026*
*Built from the seed canon, paper 113 (The Self-Organizing Spreadsheet), and the user's "take everything as far as your team is able" instruction. The substrate is the soil. The witness log is the rain. The trainer is the planting.*
