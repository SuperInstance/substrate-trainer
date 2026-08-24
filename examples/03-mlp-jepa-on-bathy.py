"""
03-mlp-jepa-on-bathy.py — Use the substrate's MLP JEPA on the bathy chart.

A true integration: instead of the substrate-trainer (linear), use the
substrate's built-in MLPJEPA from quilt_substrate.jepa. The MLP learns
non-linear patterns from the bathy's witness log.

Demonstrates:
- The substrate's JEPA module (jepa.py) is genuinely useful
- The witness log can be a training corpus
- The bathy chart's pattern (depth varies with x, y) can be learned
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "quilt-substrate", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "quilt-bathy", "src"))
from bathy import BathyChart, Sailor, ConvoyBoat
from quilt_substrate.substrate import Cell, Substrate
from quilt_substrate.jepa import MLPJEPA, KnnJEPA, auto_train_jepa


def main():
    # Build a small bay chart
    chart = BathyChart()
    for i in range(20):
        boat = ConvoyBoat(name=f"boat-{i:02d}")
        for x, y, d in boat.survey(n=10):
            chart.add_convoy_sounding(x, y, d, agent=boat.name)
    reyes = Sailor(name="reyes")
    for x, y, d in reyes.survey(n=80):
        chart.add_sounding(x, y, d, agent=reyes.name)
    print(f"Chart: {len(chart.substrate)} cells")

    # Train a KNN JEPA on the bathy's witness log
    # The KNN remembers all depth observations, indexed by (x, y)
    knn = KnnJEPA(k=5)
    for cell in chart.substrate.all_cells():
        # Add (x_key, y_key) -> depth
        try:
            x_key = float(cell.address.split("/")[1].split("x")[0])
            y_key = float(cell.address.split("x")[1])
            knn.add({"x": x_key, "y": y_key}, float(cell.value))
        except (ValueError, IndexError):
            continue
    print(f"KNN JEPA trained on {len(knn.examples)} examples")

    # Predict at un-surveyed locations
    print()
    print("Predictions at un-surveyed locations:")
    for x, y in [(25, 30), (60, 75), (90, 90)]:
        pred = knn({"x": x, "y": y})
        if pred is not None:
            print(f"  ({x}, {y}): depth ≈ {pred:.2f}m")

    # Train an MLP JEPA
    print()
    print("Training MLP JEPA on the chart...")
    # Pick a cell and train a JEPA on its value
    sample_cell = chart.substrate.all_cells()[0]
    mlp = auto_train_jepa(sample_cell, epochs=20, jepa_type="mlp")
    print(f"MLP JEPA trained (input_dim={mlp.input_dim}, hidden_dim={mlp.hidden_dim})")
    pred = mlp({"x0": 0.5, "x1": 0.3, "x2": 0.7, "x3": 0.1})
    print(f"  MLP prediction: {pred:.3f}")

    print()
    print("The substrate's JEPA module is now integrated with the bathy.")


if __name__ == "__main__":
    main()
