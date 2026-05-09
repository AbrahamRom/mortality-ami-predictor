import os
import pandas as pd
import numpy as np
from src.preprocessing import build_preprocess_pipelines
from src.features import safe_feature_columns


def test_build_preprocess_on_minimal_df(tmp_path):
    df = pd.DataFrame({
        'age': [60, 70, np.nan],
        'sex': ['M', 'F', 'F'],
        'mortality_inhospital': [0, 1, 0]
    })
    feat_cols = safe_feature_columns(df, target_cols=['mortality_inhospital'])
    X = df[feat_cols]
    y = df['mortality_inhospital']
    pipe, feat_names = build_preprocess_pipelines(X)
    pipe.fit(X, y)  # Need to fit before transform
    Xt = pipe.transform(X)
    assert Xt.shape[0] == X.shape[0]
    # Feature names are approximate since one-hot encoding changes the count
    assert Xt.shape[1] > 0
