"""Custom model persistence and loading with versioning support.

This module provides functionality to save and load custom models with:
- Model object serialization
- Metadata preservation
- Preprocessing pipeline inclusion
- Version management
- Migration support for older versions
- Validation after loading
"""

import json
import joblib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings
import hashlib
import numpy as np
import pandas as pd

from .custom_base import BaseCustomModel, CustomModelWrapper
from .metadata import ModelMetadata


# Current version of the persistence format
PERSISTENCE_VERSION = "1.0.0"


class ModelPersistenceError(Exception):
    """Exception raised for model persistence errors."""
    pass


class ModelValidationError(Exception):
    """Exception raised for model validation errors."""
    pass


def _compute_model_hash(model: Any) -> str:
    """Compute SHA256 hash of model for integrity checking.
    
    Args:
        model: Model object to hash
        
    Returns:
        Hex string of model hash
    """
    try:
        model_bytes = pickle.dumps(model)
        return hashlib.sha256(model_bytes).hexdigest()
    except Exception as e:
        warnings.warn(f"Could not compute model hash: {e}")
        return "unknown"


def save_custom_model(
    model: Union[BaseCustomModel, Any],
    path: Union[str, Path],
    metadata: Optional[ModelMetadata] = None,
    preprocessing: Optional[Any] = None,
    feature_names: Optional[list] = None,
    training_info: Optional[Dict] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Save a custom model with all associated information.
    
    Creates a directory containing:
    - model.pkl: Serialized model object
    - metadata.json: Model metadata and training info
    - preprocessing.pkl: Preprocessing pipeline (if provided)
    - manifest.json: Version info, checksums, file inventory
    
    Args:
        model: Custom model to save (BaseCustomModel or sklearn-compatible)
        path: Directory path to save model
        metadata: ModelMetadata object with model information
        preprocessing: Optional preprocessing pipeline/transformer
        feature_names: List of feature names expected by model
        training_info: Dict with training details (CV scores, hyperparams, etc.)
        overwrite: If True, overwrite existing model directory
        
    Returns:
        Dict with save information (paths, version, checksums)
        
    Raises:
        ModelPersistenceError: If save fails
        ValueError: If path exists and overwrite=False
    """
    path = Path(path)
    
    # Check if path exists
    if path.exists():
        if not overwrite:
            raise ValueError(
                f"Model directory already exists: {path}. "
                "Set overwrite=True to replace."
            )
        warnings.warn(f"Overwriting existing model at {path}")
    
    # Create directory
    path.mkdir(parents=True, exist_ok=True)
    
    save_info = {
        "version": PERSISTENCE_VERSION,
        "save_time": datetime.now().isoformat(),
        "files": {},
    }
    
    try:
        # Save model
        model_path = path / "model.pkl"
        joblib.dump(model, model_path)
        save_info["files"]["model"] = {
            "path": str(model_path),
            "size": model_path.stat().st_size,
            "hash": _compute_model_hash(model),
        }
        
        # Save preprocessing if provided
        if preprocessing is not None:
            prep_path = path / "preprocessing.pkl"
            joblib.dump(preprocessing, prep_path)
            save_info["files"]["preprocessing"] = {
                "path": str(prep_path),
                "size": prep_path.stat().st_size,
            }
        
        # Prepare metadata
        metadata_dict = {}
        if metadata is not None:
            metadata_dict = metadata.to_dict() if hasattr(metadata, 'to_dict') else {}
        
        # Add additional info
        if feature_names is not None:
            metadata_dict["feature_names"] = feature_names
        
        if training_info is not None:
            metadata_dict["training_info"] = training_info
        
        # Add model attributes if available
        if hasattr(model, 'get_params'):
            try:
                metadata_dict["model_params"] = model.get_params()
            except Exception:
                pass
        
        # Add model type info
        metadata_dict["model_class"] = type(model).__name__
        metadata_dict["model_module"] = type(model).__module__
        
        # Check if it's a BaseCustomModel
        metadata_dict["is_base_custom_model"] = isinstance(model, BaseCustomModel)
        
        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        save_info["files"]["metadata"] = {
            "path": str(metadata_path),
            "size": metadata_path.stat().st_size,
        }
        
        # Save manifest
        manifest_path = path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(save_info, f, indent=2)
        
        return save_info
        
    except Exception as e:
        # Clean up on failure
        if path.exists():
            import shutil
            shutil.rmtree(path)
        raise ModelPersistenceError(f"Failed to save model: {e}")


def load_custom_model(
    path: Union[str, Path],
    validate: bool = True,
    require_preprocessing: bool = False,
) -> Dict[str, Any]:
    """Load a custom model with all associated information.
    
    Args:
        path: Directory path containing saved model
        validate: If True, validate model after loading
        require_preprocessing: If True, raise error if no preprocessing found
        
    Returns:
        Dict containing:
            - model: Loaded model object
            - metadata: Model metadata dict
            - preprocessing: Preprocessing pipeline (None if not saved)
            - manifest: Save manifest with version info
            
    Raises:
        ModelPersistenceError: If load fails
        ModelValidationError: If validation fails
        FileNotFoundError: If required files are missing
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")
    
    result = {}
    
    try:
        # Load manifest
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                result["manifest"] = json.load(f)
        else:
            warnings.warn("No manifest.json found, loading without version info")
            result["manifest"] = {"version": "unknown"}
        
        # Check version compatibility
        version = result["manifest"].get("version", "unknown")
        if version != PERSISTENCE_VERSION and version != "unknown":
            warnings.warn(
                f"Model was saved with version {version}, "
                f"current version is {PERSISTENCE_VERSION}. "
                "Loading may fail or produce unexpected results."
            )
        
        # Load model
        model_path = path / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        result["model"] = joblib.load(model_path)
        
        # Verify model hash if available
        if "files" in result["manifest"] and "model" in result["manifest"]["files"]:
            saved_hash = result["manifest"]["files"]["model"].get("hash")
            if saved_hash and saved_hash != "unknown":
                current_hash = _compute_model_hash(result["model"])
                if current_hash != saved_hash:
                    warnings.warn(
                        "Model hash mismatch - model may have been modified or corrupted"
                    )
        
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                result["metadata"] = json.load(f)
        else:
            warnings.warn("No metadata.json found")
            result["metadata"] = {}
        
        # Load preprocessing
        prep_path = path / "preprocessing.pkl"
        if prep_path.exists():
            result["preprocessing"] = joblib.load(prep_path)
        else:
            result["preprocessing"] = None
            if require_preprocessing:
                raise FileNotFoundError(
                    f"Preprocessing pipeline required but not found at {prep_path}"
                )
        
        # Validate if requested
        if validate:
            validation_result = validate_loaded_model(
                result["model"],
                result["metadata"],
                result["preprocessing"]
            )
            result["validation"] = validation_result
            
            if not validation_result["is_valid"]:
                raise ModelValidationError(
                    f"Model validation failed: {validation_result['errors']}"
                )
        
        return result
        
    except Exception as e:
        if isinstance(e, (ModelValidationError, FileNotFoundError)):
            raise
        raise ModelPersistenceError(f"Failed to load model: {e}")


def validate_loaded_model(
    model: Any,
    metadata: Dict,
    preprocessing: Optional[Any] = None,
) -> Dict[str, Any]:
    """Validate a loaded model.
    
    Checks:
    - Model has required sklearn methods (fit, predict)
    - Model params match metadata if available
    - Preprocessing is compatible if provided
    - Feature names consistency
    
    Args:
        model: Loaded model object
        metadata: Metadata dict
        preprocessing: Optional preprocessing pipeline
        
    Returns:
        Dict with validation results:
            - is_valid: Boolean indicating if model is valid
            - errors: List of error messages
            - warnings: List of warning messages
    """
    result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
    }
    
    # Check required methods
    required_methods = ['fit', 'predict']
    for method in required_methods:
        if not hasattr(model, method):
            result["errors"].append(f"Model missing required method: {method}")
            result["is_valid"] = False
    
    # Check if it's a classifier and has predict_proba
    if metadata.get("model_type") == "classifier" or "classifier" in metadata.get("model_class", "").lower():
        if not hasattr(model, 'predict_proba'):
            result["warnings"].append(
                "Classifier model does not have predict_proba method"
            )
    
    # Validate params if available
    if "model_params" in metadata and hasattr(model, 'get_params'):
        try:
            current_params = model.get_params()
            saved_params = metadata["model_params"]
            
            # Check for parameter mismatches
            for key, saved_value in saved_params.items():
                if key in current_params:
                    current_value = current_params[key]
                    # Skip comparison for None values and objects
                    if saved_value is not None and not isinstance(saved_value, dict):
                        if str(current_value) != str(saved_value):
                            result["warnings"].append(
                                f"Parameter mismatch: {key} = {current_value} "
                                f"(saved: {saved_value})"
                            )
        except Exception as e:
            result["warnings"].append(f"Could not validate parameters: {e}")
    
    # Check preprocessing compatibility
    if preprocessing is not None:
        if not hasattr(preprocessing, 'transform'):
            result["errors"].append(
                "Preprocessing object does not have transform method"
            )
            result["is_valid"] = False
    
    # Check feature names consistency (CRITICAL FOR PREDICTION)
    if "feature_names" in metadata or metadata.get("dataset", {}).get("feature_names"):
        # Get saved features from metadata (prefer dataset.feature_names)
        saved_features = metadata.get("dataset", {}).get("feature_names") or metadata.get("feature_names", [])
        
        # Try to get feature names from model
        model_features = None
        if hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
        elif hasattr(model, 'feature_names'):
            model_features = list(model.feature_names)
        # For pipelines, check last step
        elif hasattr(model, 'steps') and len(model.steps) > 0:
            last_step = model.steps[-1][1]
            if hasattr(last_step, 'feature_names_in_'):
                model_features = list(last_step.feature_names_in_)
        
        if model_features is not None:
            # Compare feature counts
            if len(saved_features) != len(model_features):
                result["errors"].append(
                    f"CRITICAL: Feature count mismatch! "
                    f"Metadata has {len(saved_features)} features, "
                    f"but model expects {len(model_features)} features. "
                    f"Model was likely trained with different data than what's recorded in metadata."
                )
                result["is_valid"] = False
            # Compare feature names
            elif set(saved_features) != set(model_features):
                missing_in_model = set(saved_features) - set(model_features)
                extra_in_model = set(model_features) - set(saved_features)
                
                error_msg = "CRITICAL: Feature names mismatch between metadata and model!"
                if missing_in_model:
                    error_msg += f"\n  Missing in model: {list(missing_in_model)[:10]}"
                if extra_in_model:
                    error_msg += f"\n  Extra in model: {list(extra_in_model)[:10]}"
                
                result["errors"].append(error_msg)
                result["is_valid"] = False
            # Check feature order (important for some models)
            elif list(saved_features) != list(model_features):
                result["warnings"].append(
                    "Feature order differs between metadata and model. "
                    "This may cause issues with some models that depend on feature order."
                )
                # Auto-correct metadata with model's feature names
                result["warnings"].append(
                    "Auto-correcting metadata with model's feature names and order."
                )
                metadata["feature_names"] = model_features
                if "dataset" in metadata and "feature_names" in metadata["dataset"]:
                    metadata["dataset"]["feature_names"] = model_features
            else:
                # Perfect match!
                result["info"] = f"✓ Feature validation passed: {len(model_features)} features match perfectly"
        else:
            result["warnings"].append(
                "Could not extract feature_names_in_ from model for validation. "
                "Prediction may fail if dataset has incompatible features."
            )
    else:
        result["warnings"].append(
            "No feature names found in metadata. Cannot validate feature compatibility."
        )
    
    return result


def migrate_model(
    old_path: Union[str, Path],
    new_path: Union[str, Path],
    target_version: str = PERSISTENCE_VERSION,
) -> Dict[str, Any]:
    """Migrate a model from an older version to current format.
    
    Args:
        old_path: Path to old model directory
        new_path: Path to save migrated model
        target_version: Target persistence version (default: current)
        
    Returns:
        Dict with migration results
        
    Raises:
        ModelPersistenceError: If migration fails
    """
    # Load old model (without validation)
    try:
        old_data = load_custom_model(old_path, validate=False)
    except Exception as e:
        raise ModelPersistenceError(f"Failed to load old model: {e}")
    
    old_version = old_data["manifest"].get("version", "unknown")
    
    # Perform version-specific migrations
    if old_version == "unknown":
        warnings.warn("Unknown source version, attempting best-effort migration")
    
    # For now, we just re-save with current format
    # In future, add specific migration logic for each version
    try:
        save_info = save_custom_model(
            model=old_data["model"],
            path=new_path,
            metadata=old_data.get("metadata"),
            preprocessing=old_data.get("preprocessing"),
            overwrite=True,
        )
        
        return {
            "old_version": old_version,
            "new_version": target_version,
            "old_path": str(old_path),
            "new_path": str(new_path),
            "save_info": save_info,
        }
        
    except Exception as e:
        raise ModelPersistenceError(f"Failed to save migrated model: {e}")


def create_model_bundle(
    model: Any,
    X_sample: Union[np.ndarray, pd.DataFrame],
    y_sample: Optional[Union[np.ndarray, pd.Series]] = None,
    path: Union[str, Path] = None,
    **save_kwargs,
) -> Dict[str, Any]:
    """Create a complete model bundle with sample data for testing.
    
    Useful for sharing models with validation data.
    
    Args:
        model: Model to save
        X_sample: Sample input data
        y_sample: Optional sample target data
        path: Directory to save bundle
        **save_kwargs: Additional arguments for save_custom_model
        
    Returns:
        Dict with bundle information
    """
    if path is None:
        path = Path(f"model_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    path = Path(path)
    
    # Save model
    save_info = save_custom_model(model, path, **save_kwargs)
    
    # Save sample data
    sample_path = path / "sample_data.pkl"
    sample_data = {"X": X_sample}
    if y_sample is not None:
        sample_data["y"] = y_sample
    
    joblib.dump(sample_data, sample_path)
    
    save_info["bundle_info"] = {
        "sample_data_path": str(sample_path),
        "sample_size": len(X_sample),
        "n_features": X_sample.shape[1] if hasattr(X_sample, 'shape') else None,
    }
    
    return save_info


def load_model_bundle(
    path: Union[str, Path],
    test_model: bool = True,
) -> Dict[str, Any]:
    """Load a model bundle and optionally test it.
    
    Args:
        path: Path to bundle directory
        test_model: If True, run model on sample data
        
    Returns:
        Dict with model, data, and test results
    """
    path = Path(path)
    
    # Load model
    model_data = load_custom_model(path, validate=True)
    
    # Load sample data
    sample_path = path / "sample_data.pkl"
    if sample_path.exists():
        sample_data = joblib.load(sample_path)
        model_data["sample_X"] = sample_data["X"]
        model_data["sample_y"] = sample_data.get("y")
    else:
        model_data["sample_X"] = None
        model_data["sample_y"] = None
    
    # Test model if requested
    if test_model and model_data["sample_X"] is not None:
        try:
            model = model_data["model"]
            X = model_data["sample_X"]
            
            # Apply preprocessing if available
            if model_data["preprocessing"] is not None:
                X = model_data["preprocessing"].transform(X)
            
            # Make prediction
            predictions = model.predict(X)
            
            test_results = {
                "test_passed": True,
                "n_samples": len(predictions),
                "prediction_shape": predictions.shape if hasattr(predictions, 'shape') else len(predictions),
            }
            
            # Try predict_proba if available
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                test_results["has_probas"] = True
                test_results["proba_shape"] = probas.shape
            else:
                test_results["has_probas"] = False
            
            model_data["test_results"] = test_results
            
        except Exception as e:
            model_data["test_results"] = {
                "test_passed": False,
                "error": str(e),
            }
    
    return model_data


def list_saved_models(
    base_path: Union[str, Path],
    include_info: bool = True,
) -> list:
    """List all saved models in a directory.
    
    Args:
        base_path: Directory containing saved models
        include_info: If True, include metadata for each model
        
    Returns:
        List of dicts with model information
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        return []
    
    models = []
    
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if it's a valid model directory
            if (item / "model.pkl").exists():
                model_info = {
                    "path": str(item),
                    "name": item.name,
                }
                
                if include_info:
                    # Load manifest if available
                    manifest_path = item / "manifest.json"
                    if manifest_path.exists():
                        with open(manifest_path, 'r') as f:
                            model_info["manifest"] = json.load(f)
                    
                    # Load metadata if available
                    metadata_path = item / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            model_info["metadata"] = json.load(f)
                
                models.append(model_info)
    
    return models
