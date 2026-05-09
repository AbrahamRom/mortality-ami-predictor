"""Unit tests for variable_mapper module.

Tests the VariableMapper class which handles flexible column name mapping
across different clinical score types.
"""
import numpy as np
import pandas as pd
import pytest

from src.scoring.variable_mapper import VariableMapper, MappingResult, validate_mapped_dataframe


class TestVariableMapper:
    """Tests for VariableMapper class."""
    
    @pytest.fixture
    def mapper(self):
        """Create a fresh VariableMapper instance."""
        return VariableMapper()
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe with various column name formats."""
        return pd.DataFrame({
            'Age': [45, 52, 68, 55],
            'HR': [78, 85, 92, 71],
            'SBP': [145, 152, 138, 165],
            'Creatinine': [0.9, 1.2, 1.5, 0.8],
            'Killip_Class': [1, 1, 2, 1],
            'Weight': [75, 82, 68, 90],
            'Time_to_treatment': [180, 240, 120, 150],
            'muerte_inhospitalaria': [0, 1, 1, 0],
        })
    
    def test_mapper_initialization(self, mapper):
        """Test that VariableMapper initializes with expected structure."""
        assert hasattr(mapper, 'DEFAULT_ALIASES')
        assert hasattr(mapper, 'SCORE_REQUIREMENTS')
        assert 'grace' in mapper.SCORE_REQUIREMENTS
        assert 'timi_stemi' in mapper.SCORE_REQUIREMENTS
        assert 'timi_nstemi' in mapper.SCORE_REQUIREMENTS
        assert 'recuima' in mapper.SCORE_REQUIREMENTS
    
    def test_find_column_exact_match(self, mapper, sample_df):
        """Test finding a column with exact name match."""
        result = mapper.find_column(sample_df, 'age', 'grace')
        assert result == 'Age'
    
    def test_find_column_case_insensitive(self, mapper, sample_df):
        """Test finding a column with case-insensitive matching."""
        result = mapper.find_column(sample_df, 'age', 'grace')
        assert result == 'Age'
        
        result = mapper.find_column(sample_df, 'AGE', 'grace')
        assert result == 'Age'
    
    def test_find_column_with_aliases(self, mapper, sample_df):
        """Test finding a column using aliases."""
        # 'HR' is an alias for 'heart_rate'
        result = mapper.find_column(sample_df, 'heart_rate', 'grace')
        assert result == 'HR'
    
    def test_find_column_not_found(self, mapper, sample_df):
        """Test behavior when column is not found."""
        result = mapper.find_column(sample_df, 'nonexistent', 'grace')
        assert result is None
    
    def test_map_dataframe_grace(self, mapper, sample_df):
        """Test mapping dataframe columns for GRACE score."""
        result = mapper.map_dataframe(sample_df, 'grace')
        
        assert isinstance(result, MappingResult)
        assert result.success
        assert 'age' in result.mapped_df.columns
        assert 'heart_rate' in result.mapped_df.columns
        assert 'systolic_bp' in result.mapped_df.columns
        assert 'creatinine' in result.mapped_df.columns
    
    def test_map_dataframe_timi_stemi(self, mapper, sample_df):
        """Test mapping dataframe columns for TIMI-STEMI score."""
        result = mapper.map_dataframe(sample_df, 'timi_stemi')
        
        assert isinstance(result, MappingResult)
        assert 'age' in result.mapped_df.columns
        assert 'weight' in result.mapped_df.columns
        assert 'time_to_treatment' in result.mapped_df.columns
    
    def test_map_dataframe_missing_required(self, mapper):
        """Test mapping when required variables are missing."""
        df = pd.DataFrame({
            'Age': [45, 52],
            'HR': [78, 85],
            # Missing: systolic_bp, creatinine
        })
        
        result = mapper.map_dataframe(df, 'grace')
        
        assert not result.success
        assert len(result.unmapped_required) > 0
        assert 'systolic_bp' in result.unmapped_required
        assert 'creatinine' in result.unmapped_required
    
    def test_map_dataframe_partial_optional(self, mapper):
        """Test mapping when optional variables are missing."""
        df = pd.DataFrame({
            'Age': [45, 52],
            'HR': [78, 85],
            'SBP': [145, 152],
            'Creatinine': [0.9, 1.2],
            # Optional variables like prior_ecg could be missing
        })
        
        result = mapper.map_dataframe(df, 'grace')
        
        assert result.mapped_df is not None
        assert 'age' in result.mapped_df.columns
    
    def test_mapping_preserves_data(self, mapper, sample_df):
        """Test that mapping preserves data integrity."""
        result = mapper.map_dataframe(sample_df, 'grace')
        
        # Check that values are preserved
        assert result.mapped_df['age'].tolist() == sample_df['Age'].tolist()
        assert result.mapped_df['systolic_bp'].tolist() == sample_df['SBP'].tolist()
    
    def test_mapping_result_report(self, mapper):
        """Test MappingResult report generation."""
        df = pd.DataFrame({
            'Age': [45],
            'HR': [78],
            'SBP': [145],
        })
        
        result = mapper.map_dataframe(df, 'grace')
        report = result.report()
        
        assert isinstance(report, str)
        assert 'age' in report.lower()


class TestValidateMappedDataFrame:
    """Tests for validate_mapped_dataframe utility function."""
    
    def test_validate_all_columns_present(self):
        """Test validation when all required columns are present."""
        df = pd.DataFrame({
            'age': [45, 52],
            'heart_rate': [78, 85],
            'systolic_bp': [145, 152],
            'creatinine': [0.9, 1.2],
        })
        
        required_cols = ['age', 'heart_rate', 'systolic_bp', 'creatinine']
        is_valid, missing = validate_mapped_dataframe(df, required_cols)
        
        assert is_valid
        assert len(missing) == 0
    
    def test_validate_missing_columns(self):
        """Test validation when required columns are missing."""
        df = pd.DataFrame({
            'age': [45, 52],
            'heart_rate': [78, 85],
        })
        
        required_cols = ['age', 'heart_rate', 'systolic_bp', 'creatinine']
        is_valid, missing = validate_mapped_dataframe(df, required_cols)
        
        assert not is_valid
        assert 'systolic_bp' in missing
        assert 'creatinine' in missing
    
    def test_validate_null_values(self):
        """Test validation with null values in dataframe."""
        df = pd.DataFrame({
            'age': [45, np.nan],
            'heart_rate': [78, 85],
            'systolic_bp': [145, 152],
            'creatinine': [0.9, 1.2],
        })
        
        required_cols = ['age', 'heart_rate', 'systolic_bp', 'creatinine']
        is_valid, missing = validate_mapped_dataframe(df, required_cols)
        
        # Validation checks column presence, not null values
        assert is_valid
        assert len(missing) == 0


class TestVariableMapperIntegration:
    """Integration tests for VariableMapper with different score types."""
    
    @pytest.fixture
    def comprehensive_df(self):
        """Create a comprehensive dataframe with many column variations."""
        return pd.DataFrame({
            'patient_id': range(1, 11),
            'AGE': np.random.randint(30, 85, 10),
            'Heart_Rate': np.random.randint(50, 120, 10),
            'Systolic Blood Pressure': np.random.randint(90, 200, 10),
            'Creatin': np.random.uniform(0.5, 2.0, 10),
            'Sex': np.random.choice(['M', 'F'], 10),
            'Diabetes': np.random.choice([0, 1], 10),
            'Hypertension': np.random.choice([0, 1], 10),
            'CKD': np.random.choice([0, 1], 10),
            'GFR': np.random.uniform(15, 120, 10),
            'Compl': np.random.choice([0, 1, 2], 10),
            'Anterior': np.random.choice([0, 1], 10),
            'Weight_Kg': np.random.uniform(50, 120, 10),
            'Time_to_PCI': np.random.randint(60, 600, 10),
            'Killip': np.random.choice([1, 2, 3], 10),
            'ST_Change': np.random.choice([0, 1], 10),
            'Elevated_Troponin': np.random.choice([0, 1], 10),
            'Death': np.random.choice([0, 1], 10),
        })
    
    def test_all_score_types_mapping(self, comprehensive_df):
        """Test mapping for all score types with a comprehensive dataframe."""
        mapper = VariableMapper()
        
        for score_type in ['grace', 'timi_stemi', 'timi_nstemi', 'recuima']:
            result = mapper.map_dataframe(comprehensive_df, score_type)
            assert result.mapped_df is not None
            assert len(result.mapped_df) == len(comprehensive_df)
    
    def test_consistent_row_count(self, comprehensive_df):
        """Test that mapping preserves the number of rows."""
        mapper = VariableMapper()
        original_rows = len(comprehensive_df)
        
        for score_type in ['grace', 'timi_stemi', 'timi_nstemi', 'recuima']:
            result = mapper.map_dataframe(comprehensive_df, score_type)
            assert len(result.mapped_df) == original_rows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
