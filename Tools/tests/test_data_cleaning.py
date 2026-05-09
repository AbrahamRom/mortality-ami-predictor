"""Tests unitarios para el módulo de limpieza de datos."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import json

from src.cleaning import (
    CleaningConfig,
    DataCleaner,
    VariableMetadata,
    quick_clean
)


@pytest.fixture
def sample_dataframe():
    """Crea un DataFrame de prueba con varios problemas de calidad."""
    np.random.seed(42)
    
    data = {
        'numeric_normal': np.random.randn(100),
        'numeric_with_missing': [np.nan if i % 5 == 0 else np.random.rand() for i in range(100)],
        'numeric_with_outliers': np.concatenate([np.random.randn(95), [100, -100, 200, -200, 300]]),
        'categorical': ['A', 'B', 'C', None] * 25,
        'categorical_binary': ['yes', 'no'] * 50,
        'constant': ['constant'] * 100,
        'all_missing': [np.nan] * 100,
        'duplicate_col': [1, 2, 3, 4, 5] * 20,
    }
    
    df = pd.DataFrame(data)
    
    # Agregar filas duplicadas
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    return df


@pytest.fixture
def cleaning_config():
    """Configuración de limpieza por defecto."""
    return CleaningConfig(
        numeric_imputation='median',
        categorical_imputation='mode',
        outlier_method='iqr',
        outlier_treatment='cap',
        categorical_encoding='label',
        drop_duplicates=True,
        drop_fully_missing=True,
        drop_constant=True
    )


class TestVariableMetadata:
    """Tests para la clase VariableMetadata."""
    
    def test_creation(self):
        """Test de creación de metadatos."""
        meta = VariableMetadata(
            name='test_var',
            description='Test variable',
            is_numerical=True
        )
        
        assert meta.name == 'test_var'
        assert meta.description == 'Test variable'
        assert meta.is_numerical is True
        assert meta.is_categorical is False
    
    def test_to_dict(self):
        """Test de conversión a diccionario."""
        meta = VariableMetadata(
            name='test_var',
            original_type='float64',
            cleaned_type='float64'
        )
        
        meta_dict = meta.to_dict()
        
        assert isinstance(meta_dict, dict)
        assert meta_dict['name'] == 'test_var'
        assert meta_dict['original_type'] == 'float64'


class TestCleaningConfig:
    """Tests para la clase CleaningConfig."""
    
    def test_default_config(self):
        """Test de configuración por defecto."""
        config = CleaningConfig()
        
        # Defaults changed to 'none' to preserve data relationships
        assert config.numeric_imputation == 'none'
        assert config.categorical_imputation == 'none'
        assert config.outlier_method == 'none'
        assert config.outlier_treatment == 'none'
        assert config.drop_duplicates is True
    
    def test_custom_config(self):
        """Test de configuración personalizada."""
        config = CleaningConfig(
            numeric_imputation='mean',
            knn_neighbors=10,
            iqr_multiplier=2.0
        )
        
        assert config.numeric_imputation == 'mean'
        assert config.knn_neighbors == 10
        assert config.iqr_multiplier == 2.0
    
    def test_to_dict(self):
        """Test de serialización."""
        config = CleaningConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'numeric_imputation' in config_dict
        assert 'categorical_imputation' in config_dict


class TestDataCleaner:
    """Tests para la clase DataCleaner."""
    
    def test_initialization(self, cleaning_config):
        """Test de inicialización del limpiador."""
        cleaner = DataCleaner(cleaning_config)
        
        assert cleaner.config == cleaning_config
        assert cleaner.metadata == {}
        assert cleaner.encoders == {}
    
    def test_identify_column_types(self, sample_dataframe):
        """Test de identificación de tipos de columnas."""
        cleaner = DataCleaner()
        numeric_cols, categorical_cols = cleaner._identify_column_types(sample_dataframe)
        
        assert 'numeric_normal' in numeric_cols
        assert 'categorical' in categorical_cols
    
    def test_drop_fully_missing_columns(self, sample_dataframe, cleaning_config):
        """Test de eliminación de columnas totalmente vacías."""
        cleaner = DataCleaner(cleaning_config)
        
        initial_shape = sample_dataframe.shape
        cleaner.fit_transform(sample_dataframe)
        
        # La columna 'all_missing' debe ser eliminada
        assert 'all_missing' not in cleaner.metadata or \
               'fully_missing' in cleaner.metadata.get('all_missing', VariableMetadata(name='')).quality_flags
    
    def test_drop_constant_columns(self, sample_dataframe, cleaning_config):
        """Test de eliminación de columnas constantes."""
        cleaner = DataCleaner(cleaning_config)
        df_clean = cleaner.fit_transform(sample_dataframe)
        
        # La columna 'constant' debe ser eliminada
        assert 'constant' not in df_clean.columns
    
    def test_remove_duplicates(self, sample_dataframe):
        """Test de eliminación de duplicados."""
        config = CleaningConfig(drop_duplicates=True)
        cleaner = DataCleaner(config)
        
        initial_rows = len(sample_dataframe)
        df_clean = cleaner.fit_transform(sample_dataframe)
        
        # Debe haber menos filas después de eliminar duplicados
        assert len(df_clean) < initial_rows
        assert len(df_clean) == len(df_clean.drop_duplicates())
    
    def test_impute_numeric_median(self, sample_dataframe):
        """Test de imputación numérica con mediana."""
        config = CleaningConfig(
            numeric_imputation='median',
            drop_fully_missing=False,
            drop_constant=False
        )
        cleaner = DataCleaner(config)
        df_clean = cleaner.fit_transform(sample_dataframe)
        
        # No debe haber valores faltantes en columnas numéricas imputadas
        if 'numeric_with_missing' in df_clean.columns:
            assert df_clean['numeric_with_missing'].isna().sum() == 0
    
    def test_impute_numeric_mean(self):
        """Test de imputación con media."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [10, np.nan, 30, 40, 50]
        })
        
        config = CleaningConfig(numeric_imputation='mean')
        cleaner = DataCleaner(config)
        df_clean = cleaner.fit_transform(df)
        
        assert df_clean['col1'].isna().sum() == 0
        assert df_clean['col2'].isna().sum() == 0
    
    def test_outlier_detection_iqr(self, sample_dataframe):
        """Test de detección de outliers con IQR."""
        config = CleaningConfig(
            outlier_method='iqr',
            outlier_treatment='cap',
            drop_constant=False
        )
        cleaner = DataCleaner(config)
        cleaner.fit_transform(sample_dataframe)
        
        # Verificar que se detectaron outliers en la columna con outliers
        if 'numeric_with_outliers' in cleaner.metadata:
            assert cleaner.metadata['numeric_with_outliers'].outliers_detected > 0
    
    def test_outlier_treatment_cap(self):
        """Test de tratamiento de outliers por capping."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100, -100]  # 100 y -100 son outliers
        })
        
        config = CleaningConfig(
            outlier_method='iqr',
            outlier_treatment='cap',
            iqr_multiplier=1.5
        )
        cleaner = DataCleaner(config)
        df_clean = cleaner.fit_transform(df)
        
        # Los valores extremos deben estar limitados
        assert df_clean['values'].max() < 100
        assert df_clean['values'].min() > -100
    
    def test_categorical_encoding_label(self, sample_dataframe):
        """Test de codificación label para categóricas."""
        config = CleaningConfig(
            categorical_encoding='label',
            drop_constant=False,
            drop_fully_missing=False
        )
        cleaner = DataCleaner(config)
        df_clean = cleaner.fit_transform(sample_dataframe)
        
        # Las columnas categóricas deben ser numéricas ahora
        if 'categorical' in df_clean.columns:
            assert pd.api.types.is_numeric_dtype(df_clean['categorical'])
    
    def test_save_and_load_metadata(self, cleaning_config, sample_dataframe):
        """Test de guardar y cargar metadatos."""
        cleaner = DataCleaner(cleaning_config)
        df_clean = cleaner.fit_transform(sample_dataframe)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = Path(f.name)
        
        try:
            # Save metadata
            cleaner.save_metadata(temp_path)
            
            # Create new cleaner and load
            cleaner2 = DataCleaner(cleaning_config)
            cleaner2.load_metadata(temp_path)
            
            # Check metadata was loaded
            assert len(cleaner2.metadata) > 0
            for name in cleaner.metadata:
                assert name in cleaner2.metadata
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_save_config(self, cleaning_config):
        """Test de guardar configuración."""
        cleaner = DataCleaner(cleaning_config)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = Path(f.name)
        
        try:
            cleaner.save_config(temp_path)
            assert temp_path.exists()
            
            # Verificar contenido
            with open(temp_path, 'r') as f:
                config_dict = json.load(f)
            
            assert 'numeric_imputation' in config_dict
        finally:
            temp_path.unlink()
    
    def test_get_cleaning_report(self, sample_dataframe, cleaning_config):
        """Test de generación de reporte."""
        cleaner = DataCleaner(cleaning_config)
        cleaner.fit_transform(sample_dataframe)
        
        report = cleaner.get_cleaning_report()
        
        assert 'timestamp' in report
        assert 'config' in report
        assert 'variables_cleaned' in report
        assert report['variables_cleaned'] > 0
    
    def test_fit_transform_with_target(self):
        """Test de limpieza preservando columna objetivo."""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': ['A', 'B', 'C', 'D', 'E'],
            'target': [0, 1, 0, 1, 0]
        })
        
        config = CleaningConfig(numeric_imputation='mean')
        cleaner = DataCleaner(config)
        df_clean = cleaner.fit_transform(df, target_column='target')
        
        # Target debe existir y no cambiar
        assert 'target' in df_clean.columns
        assert df_clean['target'].tolist() == [0, 1, 0, 1, 0]
        
        # Target no debe estar en metadatos
        assert 'target' not in cleaner.metadata


class TestQuickClean:
    """Tests para la función quick_clean."""
    
    def test_quick_clean_basic(self, sample_dataframe):
        """Test de limpieza rápida básica."""
        df_clean, cleaner = quick_clean(sample_dataframe)
        
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(cleaner, DataCleaner)
        assert len(cleaner.metadata) > 0
    
    def test_quick_clean_with_kwargs(self, sample_dataframe):
        """Test de limpieza rápida con argumentos personalizados."""
        df_clean, cleaner = quick_clean(
            sample_dataframe,
            numeric_imputation='mean',
            outlier_method='zscore',
            categorical_encoding='onehot'
        )
        
        assert cleaner.config.numeric_imputation == 'mean'
        assert cleaner.config.outlier_method == 'zscore'
        assert cleaner.config.categorical_encoding == 'onehot'


class TestEdgeCases:
    """Tests para casos extremos."""
    
    def test_empty_dataframe(self):
        """Test con DataFrame vacío."""
        df = pd.DataFrame()
        config = CleaningConfig()
        cleaner = DataCleaner(config)
        
        # Should raise ValueError for empty dataframe
        with pytest.raises(ValueError, match="Cannot clean an empty DataFrame"):
            cleaner.fit_transform(df)
    
    def test_single_column(self):
        """Test con una sola columna."""
        df = pd.DataFrame({'col': [1, 2, 3, np.nan, 5]})
        config = CleaningConfig(numeric_imputation='median')
        cleaner = DataCleaner(config)
        
        df_clean = cleaner.fit_transform(df)
        assert len(df_clean.columns) == 1
        assert df_clean['col'].isna().sum() == 0
    
    def test_all_missing_except_one(self):
        """Test con todas las columnas vacías excepto una."""
        df = pd.DataFrame({
            'valid': [1, 2, 3, 4, 5],
            'missing1': [np.nan] * 5,
            'missing2': [np.nan] * 5
        })
        
        config = CleaningConfig(drop_fully_missing=True)
        cleaner = DataCleaner(config)
        df_clean = cleaner.fit_transform(df)
        
        assert 'valid' in df_clean.columns
        assert 'missing1' not in df_clean.columns
        assert 'missing2' not in df_clean.columns
    
    def test_all_categorical(self):
        """Test con solo variables categóricas."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 10,
            'cat2': ['X', 'Y', 'Z'] * 10
        })
        
        config = CleaningConfig(categorical_encoding='label')
        cleaner = DataCleaner(config)
        df_clean = cleaner.fit_transform(df)
        
        # Todas deben ser numéricas después de encoding
        assert pd.api.types.is_numeric_dtype(df_clean['cat1'])
        assert pd.api.types.is_numeric_dtype(df_clean['cat2'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
