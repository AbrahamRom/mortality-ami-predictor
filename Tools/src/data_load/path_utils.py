"""Utility functions for managing processed data paths and file operations.

This module provides helper functions for:
- Getting latest files by timestamp
- Managing model-specific directories
- Handling plot file overwriting
- Cleaning up old files
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import glob


def normalize_resampling_strategy(strategy: Optional[str]) -> str:
    """Normalize resampling strategy for stable filesystem keys."""
    if not strategy:
        return "none"
    return str(strategy).strip().lower().replace(" ", "_")


def get_model_combination_key(model_type: str, resampling_strategy: Optional[str]) -> str:
    """Build a stable key for model+resampling combination."""
    return f"{model_type}+{normalize_resampling_strategy(resampling_strategy)}"


def split_model_combination_key(combination_key: str) -> tuple[str, str]:
    """Split a combination key into model type and resampling strategy."""
    if "+" not in combination_key:
        return combination_key, "none"
    model_type, strategy = combination_key.split("+", 1)
    return model_type, strategy


def get_timestamp() -> str:
    """Get current timestamp in standard format YYYYMMDD_HHMMSS.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Extract timestamp from filename with format *_YYYYMMDD_HHMMSS.*.
    
    Args:
        filename: Filename containing timestamp
        
    Returns:
        datetime object if timestamp found, None otherwise
    """
    pattern = r"(\d{8}_\d{6})"
    match = re.search(pattern, filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


def get_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Get the most recent file matching pattern based on timestamp in filename.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for files (e.g., "model_dtree_*.joblib")
        
    Returns:
        Path to most recent file, or None if no files found
    """
    if not directory.exists():
        return None
    
    files = list(directory.glob(pattern))
    if not files:
        return None
    
    # Sort by timestamp extracted from filename
    files_with_timestamps = []
    for file in files:
        timestamp = extract_timestamp_from_filename(file.name)
        if timestamp:
            files_with_timestamps.append((file, timestamp))
    
    if not files_with_timestamps:
        # Fallback to modification time
        return max(files, key=lambda f: f.stat().st_mtime)
    
    # Return file with latest timestamp
    return max(files_with_timestamps, key=lambda x: x[1])[0]


def get_latest_model(
    model_type: str,
    models_dir: Path,
    resampling_strategy: Optional[str] = None,
) -> Optional[Path]:
    """Get the most recent model file for a model or model+resampling key.
    
    Args:
        model_type: Model type (e.g., "dtree", "knn", "xgb") or combination key (e.g., "xgb+smote")
        models_dir: Base models directory (e.g., Tools/processed/models)
        resampling_strategy: Optional resampling strategy for combination key generation
        
    Returns:
        Path to latest model file, or None if not found
    """
    model_key = (
        get_model_combination_key(model_type, resampling_strategy)
        if resampling_strategy is not None
        else model_type
    )

    model_dir = models_dir / model_key
    if not model_dir.exists():
        # Backward-compatible fallback:
        # if a base model type is requested, return the latest among all
        # model+strategy directories matching "{model_type}+*".
        if resampling_strategy is None and "+" not in model_type and models_dir.exists():
            candidates: list[Path] = []
            for subdir in models_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith(f"{model_type}+"):
                    latest = get_latest_file(subdir, f"model_{subdir.name}_*.joblib")
                    if latest:
                        candidates.append(latest)
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)
        return None
    
    return get_latest_file(model_dir, f"model_{model_key}_*.joblib")


def model_combination_exists(
    model_type: str,
    resampling_strategy: Optional[str],
    models_dir: Path,
) -> Optional[Path]:
    """Return latest model path when the model+resampling combination exists."""
    return get_latest_model(model_type, models_dir, resampling_strategy=resampling_strategy)


def get_latest_testset(model_type: str, testsets_dir: Path) -> Optional[Path]:
    """Get the most recent testset for a specific model type or task.
    
    Args:
        model_type: Model type (e.g., "dtree", "knn", "xgb") or task name (e.g., "mortality")
        testsets_dir: Testsets directory (e.g., Tools/processed/models/testsets)
        
    Returns:
        Path to latest testset file, or None if not found
    """
    if not testsets_dir.exists():
        return None
    
    # Try exact match first (model_type or task)
    result = get_latest_file(testsets_dir, f"testset_{model_type}_*.parquet")
    if result:
        return result
    
    # Try CSV format as fallback
    result = get_latest_file(testsets_dir, f"testset_{model_type}_*.csv")
    if result:
        return result
    
    # If not found, return the most recent testset file regardless of type (parquet first)
    all_testsets = sorted(
        testsets_dir.glob("testset_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if all_testsets:
        return all_testsets[0]
    
    # Try CSV as last resort
    all_testsets_csv = sorted(
        testsets_dir.glob("testset_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if all_testsets_csv:
        return all_testsets_csv[0]
    
    return None


def get_latest_trainset(model_type: str, testsets_dir: Path) -> Optional[Path]:
    """Get the most recent trainset for a specific model type.
    
    Args:
        model_type: Model type (e.g., "dtree", "knn", "xgb") or task name (e.g., "mortality")
        testsets_dir: Testsets directory (e.g., Tools/processed/models/testsets)
        
    Returns:
        Path to latest trainset file, or None if not found
    """
    if not testsets_dir.exists():
        return None
    
    # Try exact match first
    result = get_latest_file(testsets_dir, f"trainset_{model_type}_*.parquet")
    if result:
        return result
    
    # Try CSV as fallback
    result = get_latest_file(testsets_dir, f"trainset_{model_type}_*.csv")
    if result:
        return result
    
    # Return most recent trainset regardless of type
    all_trainsets = sorted(
        testsets_dir.glob("trainset_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if all_trainsets:
        return all_trainsets[0]
    
    # Try CSV as last resort
    all_trainsets_csv = sorted(
        testsets_dir.glob("trainset_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if all_trainsets_csv:
        return all_trainsets_csv[0]
    
    return None


def get_latest_model_by_task(task: str, models_dir: Path) -> Optional[Path]:
    """Get the most recent model for any model type trained on a specific task.
    
    Searches across all model type subdirectories (dtree, knn, xgb, etc.) and
    returns the most recently saved model. Since models are saved without task name,
    it returns the most recent model file found across all types.
    
    Args:
        task: Task name (e.g., "mortality", "arrhythmia") - currently not used in filename
        models_dir: Base models directory (e.g., Tools/processed/models)
        
    Returns:
        Path to latest model file for the task, or None if not found
    """
    all_models = []
    
    # Search in all model type subdirectories
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                # Look for all model files (format: model_{type}_{timestamp}.joblib)
                for model_file in model_dir.glob("model_*.joblib"):
                    all_models.append(model_file)
    
    if not all_models:
        return None
    
    # Return the most recent one based on modification time
    return max(all_models, key=lambda p: p.stat().st_mtime)


def get_model_metadata(model_path: Path) -> Optional[Path]:
    """Get the metadata file associated with a model.
    
    Args:
        model_path: Path to model file (.joblib)
        
    Returns:
        Path to metadata file if exists, None otherwise
    """
    metadata_path = model_path.with_suffix('.metadata.json')
    return metadata_path if metadata_path.exists() else None


def load_model_with_metadata(model_path: Path):
    """Load a model and its metadata.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Tuple of (model, metadata) where metadata is ModelMetadata or None
    """
    import joblib
    from ..models.metadata import ModelMetadata
    
    # Try loading with joblib first
    try:
        model = joblib.load(model_path)
    except Exception as e:
        # If joblib fails, try dill (for custom models)
        try:
            import dill
            with open(model_path, 'rb') as f:
                model = dill.load(f)
        except ImportError:
            # Try cloudpickle as fallback
            try:
                import cloudpickle
                with open(model_path, 'rb') as f:
                    model = cloudpickle.load(f)
            except ImportError:
                # Re-raise original exception if no alternative works
                raise e
    
    # Try to load metadata
    metadata_path = get_model_metadata(model_path)
    metadata = None
    if metadata_path:
        try:
            metadata = ModelMetadata.load(metadata_path)
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
    
    return model, metadata


def get_latest_plot(plots_dir: Path, base_name: str) -> Optional[Path]:
    """Get the most recent plot file (PNG or HTML) matching a base name.
    
    Args:
        plots_dir: Directory containing plots
        base_name: Base name of the plot (e.g., "comparison_matrix", "roc_curve")
        
    Returns:
        Path to latest plot file (.png or .html), or None if not found
    """
    if not plots_dir.exists():
        return None
    
    # Try PNG first
    png_file = get_latest_file(plots_dir, f"{base_name}_*.png")
    if png_file:
        return png_file
    
    # Try exact match PNG
    png_exact = plots_dir / f"{base_name}.png"
    if png_exact.exists():
        return png_exact
    
    # Try HTML
    html_file = get_latest_file(plots_dir, f"{base_name}_*.html")
    if html_file:
        return html_file
    
    # Try exact match HTML
    html_exact = plots_dir / f"{base_name}.html"
    if html_exact.exists():
        return html_exact
    
    return None


def get_latest_cleaned_dataset(cleaned_datasets_dir: Path) -> Optional[Path]:
    """Get the most recent cleaned dataset.
    
    Args:
        cleaned_datasets_dir: Cleaned datasets directory
        
    Returns:
        Path to latest cleaned dataset, or None if not found
    """
    return get_latest_file(cleaned_datasets_dir, "cleaned_dataset_*.csv")


def save_plot_with_overwrite(
    fig,
    plot_dir: Path,
    plot_name: str,
    model_type: Optional[str] = None,
    formats: List[str] = None
) -> List[Path]:
    """Save a plot, overwriting previous version for the same model type.
    
    This function ensures only the latest plot of each type is kept for each model.
    
    Args:
        fig: Plotly or matplotlib figure object
        plot_dir: Base plots directory (e.g., Tools/processed/plots/evaluation)
        plot_name: Base name for the plot (e.g., "roc_curve", "confusion_matrix")
        model_type: Model type (e.g., "dtree"). If None, saves without model subdirectory
        formats: List of formats to save (default: ["png", "html"])
        
    Returns:
        List of paths where plots were saved
    """
    if formats is None:
        formats = ["png", "html"]
    
    # Determine save directory
    if model_type:
        save_dir = plot_dir / model_type
    else:
        save_dir = plot_dir
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove old versions of this plot for this model
    old_files = list(save_dir.glob(f"{plot_name}.*"))
    for old_file in old_files:
        try:
            old_file.unlink()
        except Exception:
            pass  # Ignore errors if file is in use
    
    # Save new plot
    saved_paths = []
    for fmt in formats:
        save_path = save_dir / f"{plot_name}.{fmt}"
        
        try:
            if fmt == "html":
                # Plotly figure
                if hasattr(fig, 'write_html'):
                    fig.write_html(str(save_path))
                    saved_paths.append(save_path)
            elif fmt == "png":
                # Try plotly first, then matplotlib
                if hasattr(fig, 'write_image'):
                    fig.write_image(str(save_path))
                    saved_paths.append(save_path)
                elif hasattr(fig, 'savefig'):
                    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
                    saved_paths.append(save_path)
            elif fmt in ["jpg", "jpeg", "svg", "pdf"]:
                if hasattr(fig, 'write_image'):
                    fig.write_image(str(save_path))
                    saved_paths.append(save_path)
                elif hasattr(fig, 'savefig'):
                    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
                    saved_paths.append(save_path)
        except Exception as e:
            print(f"Warning: Could not save plot in {fmt} format: {e}")
    
    return saved_paths


def save_model_with_cleanup(
    model,
    model_type: str,
    models_dir: Path,
    keep_n_latest: int = 1,
    metadata = None,
    training_data = None,
    combination_key: Optional[str] = None,
) -> Path:
    """Save model with metadata and optionally clean up old versions.
    
    Args:
        model: Model object to save (must be joblib-serializable)
        model_type: Model type (e.g., "dtree", "knn", "xgb")
        models_dir: Base models directory (e.g., Tools/processed/models)
        keep_n_latest: Number of latest models to keep (default: 1)
        metadata: Optional ModelMetadata object to save alongside the model
        training_data: Optional DataFrame with the EXACT data used for training (after feature selection)
        combination_key: Optional model+resampling key (e.g., "xgb+smote")
        
    Returns:
        Path where model was saved
    """
    import joblib
    
    model_key = combination_key or model_type

    model_dir = models_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save new model
    timestamp = get_timestamp()
    save_path = model_dir / f"model_{model_key}_{timestamp}.joblib"
    
    # Check if model contains custom classes (which may have pickle issues)
    # Try to detect custom models by checking the module
    try:
        from src.models.custom_base import BaseCustomModel, BaseCustomClassifier, BaseCustomRegressor
        
        # Check if the model or any component is a custom model
        is_custom = False
        if hasattr(model, 'steps'):  # Pipeline
            for name, component in model.steps:
                if isinstance(component, (BaseCustomClassifier, BaseCustomRegressor)):
                    is_custom = True
                    break
        elif isinstance(model, (BaseCustomClassifier, BaseCustomRegressor)):
            is_custom = True
        
        if is_custom:
            # Use dill for custom models (handles dynamic classes better)
            try:
                import dill
                with open(save_path, 'wb') as f:
                    dill.dump(model, f)
            except ImportError:
                # Fallback to cloudpickle
                try:
                    import cloudpickle
                    with open(save_path, 'wb') as f:
                        cloudpickle.dump(model, f)
                except ImportError:
                    # Last resort: try joblib with protocol 4
                    joblib.dump(model, save_path, protocol=4)
        else:
            # Standard joblib for regular models
            joblib.dump(model, save_path)
    except ImportError:
        # If custom model base classes not available, use joblib
        joblib.dump(model, save_path)
    
    # Save metadata if provided
    if metadata is not None:
        metadata_path = model_dir / f"model_{model_key}_{timestamp}.metadata.json"
        metadata.model_file_path = str(save_path)
        metadata.save(metadata_path, format="json")
    
    # Save training data if provided (this is the ACTUAL data used to train, with selected features)
    if training_data is not None:
        training_data_path = model_dir / f"training_data_{model_key}_{timestamp}.parquet"
        try:
            training_data.to_parquet(training_data_path, index=False)
        except Exception as e:
            # Fallback to CSV if parquet fails
            training_data_path = model_dir / f"training_data_{model_key}_{timestamp}.csv"
            training_data.to_csv(training_data_path, index=False)
    
    # Clean up old models, metadata, and training data
    if keep_n_latest > 0:
        old_models = sorted(
            model_dir.glob(f"model_{model_key}_*.joblib"),
            key=lambda f: extract_timestamp_from_filename(f.name) or datetime.min,
            reverse=True
        )
        
        # Keep only the N most recent
        for old_model in old_models[keep_n_latest:]:
            try:
                old_model.unlink()
                # Also delete associated metadata file
                metadata_file = old_model.with_suffix('.metadata.json')
                if metadata_file.exists():
                    metadata_file.unlink()
                # Also delete associated training data
                ts = extract_timestamp_from_filename(old_model.name)
                if ts:
                    for ext in ['parquet', 'csv']:
                        training_data_file = model_dir / f"training_data_{model_key}_{ts}.{ext}"
                        if training_data_file.exists():
                            training_data_file.unlink()
            except Exception:
                pass  # Ignore errors
    
    return save_path


def save_dataset_with_timestamp(
    df,
    directory: Path,
    prefix: str = "dataset",
    format: str = "csv"
) -> Path:
    """Save dataset with timestamp.
    
    Args:
        df: pandas DataFrame
        directory: Directory to save to
        prefix: Filename prefix (e.g., "cleaned_dataset", "testset_dtree")
        format: File format ("csv" or "parquet")
        
    Returns:
        Path where dataset was saved
    """
    directory.mkdir(parents=True, exist_ok=True)
    
    timestamp = get_timestamp()
    filename = f"{prefix}_{timestamp}.{format}"
    save_path = directory / filename
    
    if format == "csv":
        df.to_csv(save_path, index=False)
    elif format == "parquet":
        df.to_parquet(save_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return save_path


def cleanup_old_testsets(
    model_type: str,
    testsets_dir: Path,
    keep_n_latest: int = 1
) -> None:
    """Clean up old testsets for a model type, keeping only the N most recent.
    
    IMPORTANT: This function also checks if testsets are referenced in existing
    model metadata files and will NOT delete those, even if they're old.
    
    Args:
        model_type: Model type (e.g., "dtree", "knn", "xgb")
        testsets_dir: Testsets directory
        keep_n_latest: Number of latest testsets to keep (default: 1)
    """
    import json
    from pathlib import Path
    
    # Get all metadata files to find referenced testsets
    protected_testsets = set()
    protected_trainsets = set()
    
    # Search for metadata files in model directories
    models_base_dir = testsets_dir.parent  # processed/models
    for model_dir in models_base_dir.iterdir():
        if model_dir.is_dir() and model_dir.name not in ['testsets', '__pycache__']:
            # Check all metadata files in this model directory
            for metadata_file in model_dir.glob("*.metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Get test and train set paths from metadata
                    dataset_info = metadata.get('dataset', {})
                    test_path = dataset_info.get('test_set_path')
                    train_path = dataset_info.get('train_set_path')
                    
                    if test_path:
                        protected_testsets.add(Path(test_path).name)
                    if train_path:
                        protected_trainsets.add(Path(train_path).name)
                except Exception:
                    pass  # Ignore errors reading metadata
    
    # Now clean up testsets/trainsets
    for prefix, protected_set in [("testset", protected_testsets), ("trainset", protected_trainsets)]:
        old_files = sorted(
            testsets_dir.glob(f"{prefix}_{model_type}_*.parquet"),
            key=lambda f: extract_timestamp_from_filename(f.name) or datetime.min,
            reverse=True
        )
        
        # Filter out protected files
        candidates_for_deletion = []
        for f in old_files[keep_n_latest:]:
            if f.name not in protected_set:
                candidates_for_deletion.append(f)
        
        # Delete non-protected old files
        for old_file in candidates_for_deletion:
            try:
                old_file.unlink()
            except Exception:
                pass


def get_all_model_types(models_dir: Path) -> List[str]:
    """Get list of all model directories that contain saved models.
    
    Args:
        models_dir: Base models directory
        
    Returns:
        List of model directory names (includes combination keys like "xgb+smote")
    """
    if not models_dir.exists():
        return []
    
    model_types = []
    for subdir in models_dir.iterdir():
        if subdir.is_dir() and subdir.name != "testsets":
            # Check if directory has any model files
            if list(subdir.glob("model_*.joblib")):
                model_types.append(subdir.name)
    
    return sorted(model_types)


def get_plots_dir(plot_type: str) -> Path:
    """Get the path to a specific plots subdirectory.
    
    Args:
        plot_type: Type of plots ('eda', 'evaluation', 'explainability', 'training')
        
    Returns:
        Path to the plots subdirectory
    """
    from ..config import CONFIG
    
    plots_dir = Path(CONFIG.plots_dir)
    plots_subdir = plots_dir / plot_type
    plots_subdir.mkdir(parents=True, exist_ok=True)
    
    return plots_subdir
