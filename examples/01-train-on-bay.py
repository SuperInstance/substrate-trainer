"""
01-train-on-bay.py — Train a JEPA model on the bay's witness log.

Scenario 03 (The Convoy) implemented as a trainer.
100 cells, 3 agents (reyes, skate, inference), 10 epochs of training.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "quilt-substrate", "src"))
from substrate import Substrate, Cell
from trainer import Trainer


def main():
    # Build a substrate: 100 cells, 3 agents, lots of witness entries
    s = Substrate()
    n_cells = 100
    agents = ["reyes", "skate", "inference"]
    for i in range(n_cells):
        # Cells with values 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, ... cycling
        c = Cell(address=f"bay/{i:03d}", value=10.0 + (i % 5) * 0.5)
        s.add(c)
        for agent in agents:
            s.witness(c, agent, "read", c.value)
    print(f"Built substrate: {len(s)} cells, witness log: {sum(len(c.witness_log) for c in s.all_cells())} entries")

    # Train
    trainer = Trainer()
    model = trainer.fit(s, n_epochs=20)
    print(f"Trained: {model.n_train} examples, {len(model.agent_to_id)} agents, target mean={model.target_mean:.3f}, std={model.target_std:.3f}")

    # Predict
    for ctx in [["reyes"], ["reyes", "skate"], ["reyes", "skate", "inference"]]:
        pred, conf = model.predict(ctx)
        print(f"  Predict({ctx}): {pred:.3f} (confidence {conf:.3f})")

    # Compare to a specific cell
    cell = s.get("bay/050")
    print(f"  Actual value of bay/050: {cell.value}, confidence: {cell.confidence:.3f}")


if __name__ == "__main__":
    main()
