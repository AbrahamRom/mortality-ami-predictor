"""Tests for the new modular structure."""
import pytest
import pandas as pd
import numpy as np

# Test data module
def test_data_loaders():
    """Test data loading functionality."""
    from src.data_load import load_dataset
    
    # Create a temporary CSV
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        
        # Test loading
        loaded = load_dataset(f.name)
        assert loaded.shape == (3, 2)
        assert list(loaded.columns) == ['a', 'b']


def test_data_splitters():
    """Test data splitting functionality."""
    from src.data_load import train_test_split
    
    df = pd.DataFrame({
        'feature': range(100),
        'target': [0, 1] * 50
    })
    
    # Test stratified split (uses stratify_column parameter)
    train, test = train_test_split(
        df, 
        test_size=0.2, 
        stratify_column='target'
    )
    
    assert len(train) == 80
    assert len(test) == 20


# Test cleaning module
def test_cleaning_quick_clean():
    """Test quick_clean functionality."""
    from src.cleaning import quick_clean
    
    df = pd.DataFrame({
        'num': [1.0, 2.0, np.nan, 4.0],
        'cat': ['a', 'b', None, 'a'],
        'target': [0, 1, 0, 1]
    })
    
    # quick_clean returns (cleaned_df, cleaner)
    cleaned, cleaner = quick_clean(df, target_column='target')
    
    # Should have no missing values
    assert cleaned.isna().sum().sum() == 0


# Test features module
def test_features_safe_columns():
    """Test safe_feature_columns."""
    from src.features import safe_feature_columns
    
    df = pd.DataFrame({
        'patient_id': [1, 2, 3],
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0],
        'target': [0, 1, 0]
    })
    
    # safe_feature_columns uses target_cols parameter (list)
    features = safe_feature_columns(df, target_cols=['target'])
    
    assert 'feature1' in features
    assert 'feature2' in features
    assert 'target' not in features
    # 'patient_id' is auto-excluded by EXCLUDE_COLS
    assert 'patient_id' not in features


# Test preprocessing module
def test_preprocessing_pipeline():
    """Test preprocessing pipeline building."""
    from src.preprocessing import build_preprocessing_pipeline
    
    df = pd.DataFrame({
        'num1': [1.0, 2.0, 3.0],
        'num2': [4.0, 5.0, 6.0],
        'cat1': ['a', 'b', 'a']
    })
    
    # build_preprocessing_pipeline returns (pipeline, feature_names)
    pipeline, feature_names = build_preprocessing_pipeline(df)
    
    # Should be able to transform
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 3


# Test models module
def test_models_classifiers():
    """Test classifier creation."""
    from src.models import make_classifiers
    
    classifiers = make_classifiers()
    
    assert 'knn' in classifiers
    assert 'logreg' in classifiers  # actual key is 'logreg' not 'logistic'
    assert 'dtree' in classifiers  # actual key is 'dtree' not 'decision_tree'


def test_models_get_model():
    """Test model registry."""
    from src.models import get_model
    
    # Use correct model name: 'logreg' not 'logistic'
    model = get_model('logreg', task='classification')
    assert model is not None


# Test evaluation module
def test_evaluation_metrics():
    """Test metrics computation."""
    from src.evaluation import compute_classification_metrics
    
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
    
    metrics = compute_classification_metrics(y_true, y_prob)
    
    assert 'auroc' in metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 0 <= metrics['auroc'] <= 1


# Test scoring module
def test_scoring_grace():
    """Test GRACE score calculation."""
    from src.scoring import get_score
    
    grace = get_score('grace')
    result = grace.compute(
        age=65,
        heart_rate=85,
        systolic_bp=120,
        creatinine=1.0,
        cardiac_arrest=False,
        st_deviation=True,
        elevated_enzymes=False,
        killip_class=1
    )
    
    assert 'score' in result
    assert 'risk_category' in result
    assert isinstance(result['score'], float)


# Test explainability module (optional - requires shap)
def test_explainability_permutation():
    """Test permutation importance."""
    from src.explainability import compute_permutation_importance
    from sklearn.ensemble import RandomForestClassifier
    
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    importance = compute_permutation_importance(
        model, X, y,
        feature_names=[f'f{i}' for i in range(5)],
        n_repeats=3
    )
    
    assert 'importances_mean' in importance
    assert len(importance['importances_mean']) == 5


# Test prediction module
def test_prediction_predictor():
    """Test predictor wrapper."""
    from src.prediction import Predictor
    from sklearn.ensemble import RandomForestClassifier
    
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    predictor = Predictor(model, model_name='test')
    
    # Test prediction
    X_test = np.random.rand(10, 5)
    predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test)
    
    assert len(predictions) == 10
    assert len(probabilities) == 10


# Test EDA module
def test_eda_analyzer():
    """Test EDA analyzer."""
    from src.eda import EDAAnalyzer
    
    df = pd.DataFrame({
        'num1': np.random.rand(100),
        'num2': np.random.rand(100),
        'cat1': np.random.choice(['a', 'b', 'c'], 100)
    })
    
    analyzer = EDAAnalyzer(df)
    
    # Test univariate analysis
    results = analyzer.analyze_univariate()
    assert len(results) == 3
    assert 'num1' in results
    assert 'cat1' in results


def test_eda_quick_eda():
    """Test quick_eda function."""
    from src.eda import quick_eda
    
    df = pd.DataFrame({
        'num1': np.random.rand(50),
        'num2': np.random.rand(50),
    })
    
    analyzer = quick_eda(df, run_pca=False)
    
    assert len(analyzer.univariate_results) > 0
    
    # Test with PCA
    analyzer_pca = quick_eda(df, run_pca=True)
    assert analyzer_pca.pca_results is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
