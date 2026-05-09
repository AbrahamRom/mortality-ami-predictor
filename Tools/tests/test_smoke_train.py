import os
import pandas as pd
import numpy as np
from src.training import fit_and_save_best_classifier


def test_smoke_train_small(tmp_path, monkeypatch):
    # Small synthetic dataset
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        'age': rng.integers(40, 90, size=100),
        'sbp': rng.normal(120, 15, size=100),
        'sex': rng.choice(['M','F'], size=100),
    })
    y = rng.integers(0, 2, size=100)

    path, model = fit_and_save_best_classifier(X, pd.Series(y), quick=True, task_name='test')
    assert os.path.exists(path)
