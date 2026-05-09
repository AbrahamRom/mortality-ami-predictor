import os
import pandas as pd
import numpy as np
import joblib

from src.models import make_classifiers
from src.preprocessing import build_preprocess_pipelines


def test_model_save_load(tmp_path):
    X = pd.DataFrame({'x1': [0,1,2,3,4,5], 'x2': ['a','b','a','b','a','b']})
    y = np.array([0,1,0,1,0,1])

    pre, _ = build_preprocess_pipelines(X)
    name, (clf, grid) = next(iter(make_classifiers().items()))

    from sklearn.pipeline import Pipeline
    pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])
    pipe.fit(X, y)

    path = tmp_path / 'model.joblib'
    joblib.dump(pipe, path)
    loaded = joblib.load(path)

    preds = loaded.predict_proba(X)[:,1]
    assert preds.shape[0] == len(y)
