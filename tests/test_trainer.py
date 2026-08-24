"""Tests for substrate-trainer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "quilt-substrate", "src"))
from substrate import Substrate, Cell
from trainer import WitnessLogDataset, JEPAModel, Trainer


def test_dataset_builds_from_substrate():
    s = Substrate()
    for i in range(10):
        c = Cell(address=f"bay/{i:03d}", value=12.5 + (i % 5))
        s.add(c)
        for agent in ["reyes", "skate"]:
            s.witness(c, agent, "read", c.value)
    dataset = WitnessLogDataset(s).build()
    assert len(dataset) > 0


def test_dataset_uses_cell_values():
    s = Substrate()
    c = Cell(address="bay/001", value=42.0)
    s.add(c)
    s.witness(c, "reyes", "read", 42.0)
    s.witness(c, "reyes", "read", 42.0)
    dataset = WitnessLogDataset(s).build()
    assert all(pair.target_value == 42.0 for pair in dataset.pairs)


def test_dataset_respects_max_context():
    s = Substrate()
    c = Cell(address="bay/001", value=42.0)
    s.add(c)
    for agent in ["reyes", "skate", "inference", "drone", "boat", "fish"]:
        s.witness(c, agent, "read", 42.0)
    dataset = WitnessLogDataset(s, max_context=3).build()
    for pair in dataset.pairs:
        assert len(pair.context_addresses) <= 3


def test_jepa_model_predicts():
    model = JEPAModel()
    pred, conf = model.predict(["reyes"])
    assert isinstance(pred, float)
    assert 0.0 <= conf <= 1.0


def test_jepa_model_fit_reduces_loss():
    s = Substrate()
    for i in range(20):
        c = Cell(address=f"bay/{i:03d}", value=10.0 + (i % 3) * 0.5)
        s.add(c)
        for agent in ["reyes", "skate"]:
            s.witness(c, agent, "read", c.value)
    dataset = WitnessLogDataset(s).build()
    model = JEPAModel()
    # Initial loss
    initial_loss = sum((model.predict(p.context_addresses)[0] - p.target_value) ** 2 for p in dataset.pairs)
    model.fit(dataset, n_epochs=20)
    final_loss = sum((model.predict(p.context_addresses)[0] - p.target_value) ** 2 for p in dataset.pairs)
    assert final_loss < initial_loss


def test_trainer_returns_model():
    s = Substrate()
    c = Cell(address="bay/001", value=42.0)
    s.add(c)
    s.witness(c, "reyes", "read", 42.0)
    s.witness(c, "skate", "read", 42.0)
    trainer = Trainer()
    model = trainer.fit(s, n_epochs=5)
    assert isinstance(model, JEPAModel)
    assert model.n_train > 0


def test_fable_18_spyglass_and_convoy_trainer_sees_convoy():
    """Fable 18: the trainer is trained on the convoy's data."""
    s = Substrate()
    c = Cell(address="bay/A17", value=12.5)
    s.add(c)
    for i in range(50):
        s.witness(c, f"boat-{i:03d}", "read", 12.5)
        s.witness(c, "reyes", "read", 12.5)  # need a second agent for context
    trainer = Trainer()
    model = trainer.fit(s, n_epochs=10)
    # The model should have seen multiple agents
    assert len(model.agent_to_id) > 1


def test_model_predicts_close_to_truth_after_training():
    """After training, the model should predict close to the actual value."""
    s = Substrate()
    # All cells have value 42.0
    for i in range(30):
        c = Cell(address=f"bay/{i:03d}", value=42.0)
        s.add(c)
        s.witness(c, "reyes", "read", 42.0)
        s.witness(c, "skate", "read", 42.0)
    trainer = Trainer()
    model = trainer.fit(s, n_epochs=50)
    # Predict with the same context
    pred, conf = model.predict(["reyes", "skate"])
    # Should be close to 42.0
    assert abs(pred - 42.0) < 5.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
