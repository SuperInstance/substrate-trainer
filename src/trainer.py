"""
substrate-trainer — Train JEPA-like models on the substrate's witness log.

The substrate is the soil. The model is the plant. The witness log is the rain.

The trainer takes a substrate's witness log and builds a training set of
(cell_context, next_value) pairs. It trains a JEPA-like model that predicts
missing cells from their context.

The model is *substrate-native*:
- embeddings are the cells' tensors
- attention is over the cells' connections
- predictions are the cells' values
- loss is the substrate's decay-weighted error
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import math
import random
import sys
import os

# We depend on quilt-substrate. Try to import it; if not available, use a fallback.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "quilt-substrate", "src"))
    from quilt_substrate.substrate import Substrate, Cell
    HAS_SUBSTRATE = True
except ImportError:
    HAS_SUBSTRATE = False
    Substrate = None
    Cell = None


# -- The training set ------------------------------------------------------

@dataclass
class TrainingPair:
    """A (context, target) pair for the JEPA model."""
    context_addresses: List[str]  # the addresses of the context cells
    target_address: str  # the address of the target cell
    target_value: float  # the target's value
    confidence: float  # the target's confidence (decay-weighted)


class WitnessLogDataset:
    """Build training pairs from a substrate's witness log.

    For each (cell, agent) pair in the witness log:
    - The context is the set of cells the agent read *just before* this cell
    - The target is the value of this cell
    - The confidence is the cell's current confidence
    """

    def __init__(self, substrate, max_context: int = 8, lookback: int = 5):
        self.substrate = substrate
        self.max_context = max_context
        self.lookback = lookback
        self.pairs: List[TrainingPair] = []

    def build(self) -> "WitnessLogDataset":
        """Build the training set by walking the witness log."""
        if not HAS_SUBSTRATE:
            return self
        # For each cell, walk the witness log in time order
        for cell in self.substrate.all_cells():
            log = cell.witness_log
            if len(log) < 2:
                continue
            # For each read/write entry, the context is the previous reads
            for i, entry in enumerate(log[1:], start=1):
                # The context is the lookback previous entries' agents
                context_addresses = []
                for prev in log[max(0, i - self.lookback):i]:
                    # The context cell is the one the prev entry was about
                    if prev.agent_id not in context_addresses:
                        context_addresses.append(prev.agent_id)
                if not context_addresses:
                    continue
                context_addresses = context_addresses[:self.max_context]
                # The target value is the cell's current value (as a float)
                try:
                    target_value = float(cell.value)
                except (TypeError, ValueError):
                    continue
                self.pairs.append(TrainingPair(
                    context_addresses=context_addresses,
                    target_address=cell.address,
                    target_value=target_value,
                    confidence=cell.confidence,
                ))
        return self

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> TrainingPair:
        return self.pairs[idx]


# -- The JEPA-like model ---------------------------------------------------

@dataclass
class JEPAModel:
    """A simple JEPA-like model: linear attention over context, predict target.

    The model has:
    - An embedding for each agent_id (the "context agent")
    - A linear attention over the context agents
    - A linear layer to predict the target value
    - A confidence output (the model's own confidence in the prediction)
    """

    agent_to_id: Dict[str, int] = field(default_factory=dict)
    embeddings: Dict[int, float] = field(default_factory=dict)  # agent_id → embedding
    weights: List[float] = field(default_factory=lambda: [0.5])  # attention weights
    bias: float = 0.0  # output bias
    target_mean: float = 0.0  # target value mean (for normalization)
    target_std: float = 1.0  # target value std (for normalization)
    n_train: int = 0  # number of training examples seen

    def __post_init__(self):
        if not self.embeddings:
            # Default: each agent has an embedding of 0.0
            pass

    def _get_agent_id(self, agent_id: str) -> int:
        if agent_id not in self.agent_to_id:
            self.agent_to_id[agent_id] = len(self.agent_to_id)
            self.embeddings[self.agent_to_id[agent_id]] = 0.0
        return self.agent_to_id[agent_id]

    def _get_embedding(self, agent_id: str) -> float:
        idx = self._get_agent_id(agent_id)
        return self.embeddings.get(idx, 0.0)

    def predict(self, context_addresses: List[str]) -> Tuple[float, float]:
        """Predict the target value given the context.

        Returns (predicted_value, model_confidence).
        """
        if not context_addresses:
            return self.target_mean, 0.0
        # Sum the embeddings of the context agents (with attention weights)
        total = 0.0
        for i, addr in enumerate(context_addresses):
            emb = self._get_embedding(addr)
            w = self.weights[i] if i < len(self.weights) else self.weights[-1]
            total += emb * w
        total += self.bias
        # Denormalize
        predicted = total * self.target_std + self.target_mean
        # Confidence: how often the agent has been seen
        confidence = min(1.0, self.n_train / 100.0)
        return predicted, confidence

    def fit(self, dataset: WitnessLogDataset, n_epochs: int = 10, lr: float = 0.01) -> "JEPAModel":
        """Fit the model to the dataset.

        Simple gradient descent: for each (context, target) pair,
        predict, compute the loss, update the embeddings and weights.
        """
        if len(dataset) == 0:
            return self
        # Compute target mean and std
        values = [p.target_value for p in dataset.pairs]
        self.target_mean = sum(values) / len(values)
        self.target_std = max(1e-6, (sum((v - self.target_mean) ** 2 for v in values) / len(values)) ** 0.5)
        # Initialize weights for the maximum context size
        max_ctx = max(len(p.context_addresses) for p in dataset.pairs)
        while len(self.weights) < max_ctx:
            self.weights.append(0.5)
        # Train
        for epoch in range(n_epochs):
            random.shuffle(dataset.pairs)
            total_loss = 0.0
            for pair in dataset.pairs:
                # Get or create agent ids
                agent_ids = [self._get_agent_id(a) for a in pair.context_addresses]
                # Forward pass
                pred, _ = self.predict(pair.context_addresses)
                # Loss: squared error, weighted by (1 - confidence) (decay-weighted)
                weight = 1.0 - pair.confidence
                err = pred - pair.target_value
                loss = err * err * weight
                total_loss += loss
                # Backward pass (gradient descent on embeddings and weights)
                # d(loss)/d(pred) = 2 * err * weight
                # d(pred)/d(w_i) = embedding_i * target_std
                # d(pred)/d(emb_i) = w_i * target_std
                grad_pred = 2.0 * err * weight
                for i, agent_id in enumerate(agent_ids):
                    emb = self.embeddings[agent_id]
                    # Update weight
                    w = self.weights[i]
                    self.weights[i] -= lr * grad_pred * emb * self.target_std
                    # Update embedding
                    self.embeddings[agent_id] -= lr * grad_pred * w * self.target_std
                # Update bias
                self.bias -= lr * grad_pred * self.target_std
            self.n_train += len(dataset.pairs)
        return self


# -- The trainer -----------------------------------------------------------

class Trainer:
    """The substrate-trainer orchestrator.

    Takes a substrate, builds the witness-log dataset, fits the JEPA model.
    """

    def __init__(self, max_context: int = 8, lookback: int = 5, lr: float = 0.01):
        self.max_context = max_context
        self.lookback = lookback
        self.lr = lr

    def fit(self, substrate, n_epochs: int = 10) -> JEPAModel:
        """Fit a JEPA model to the substrate's witness log."""
        dataset = WitnessLogDataset(substrate, max_context=self.max_context, lookback=self.lookback)
        dataset.build()
        model = JEPAModel()
        model.fit(dataset, n_epochs=n_epochs, lr=self.lr)
        return model


# -- CLI ------------------------------------------------------------------

def _cli():
    import argparse
    p = argparse.ArgumentParser(prog="substrate-trainer", description="Train JEPA-like models on substrate witness logs.")
    sub = p.add_subparsers(dest="cmd")
    demo = sub.add_parser("demo", help="Run a small demo: build a substrate, train a model, predict.")
    demo.add_argument("--n-cells", type=int, default=50)
    demo.add_argument("--n-epochs", type=int, default=20)
    args = p.parse_args()

    if args.cmd == "demo":
        if not HAS_SUBSTRATE:
            print("quilt-substrate not installed. Run from the substrate-trainer directory.")
            return
        s = Substrate()
        for i in range(args.n_cells):
            c = Cell(address=f"bay/{i:03d}", value=12.5 + (i % 5) * 0.5)
            s.add(c)
            for agent in ["reyes", "skate", "inference"]:
                s.witness(c, agent, "read", c.value)
        print(f"Built substrate with {len(s)} cells, witness log: {sum(len(c.witness_log) for c in s.all_cells())} entries")
        trainer = Trainer()
        model = trainer.fit(s, n_epochs=args.n_epochs)
        print(f"Trained JEPA model: {model.n_train} examples, {len(model.agent_to_id)} agents seen")
        # Predict a missing cell
        pred, conf = model.predict(["reyes", "skate"])
        print(f"Prediction for context ['reyes', 'skate']: {pred:.3f} (confidence {conf:.3f})")
    else:
        p.print_help()


if __name__ == "__main__":
    _cli()
