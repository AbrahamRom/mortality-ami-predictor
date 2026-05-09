"""Unit tests for orchestrator subgroup analysis functionality.

Tests the subgroup-specific comparison logic and STEMI/NSTEMI separation
in ClinicalScaleMLOrchestrator.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, MagicMock

from src.evaluation.scale_ml_orchestrator import (
    ClinicalScaleMLOrchestrator,
    SynchronizationReport,
)


class TestSynchronizationReport:
    """Tests for SynchronizationReport dataclass."""
    
    def test_sync_report_initialization(self):
        """Test creating a SynchronizationReport."""
        report = SynchronizationReport(
            original_size=1000,
            synchronized_size=950,
            samples_removed=50,
            removal_rate=0.05,
        )
        
        assert report.original_size == 1000
        assert report.synchronized_size == 950
        assert report.samples_removed == 50
        assert report.removal_rate == 0.05
        assert report.all_methods_valid
    
    def test_sync_report_method_validity(self):
        """Test adding method validity information."""
        report = SynchronizationReport(
            original_size=1000,
            synchronized_size=950,
            samples_removed=50,
            removal_rate=0.05,
        )
        
        report.method_validity = {
            'ML Model': {'valid_count': 1000, 'missing_count': 0},
            'GRACE': {'valid_count': 970, 'missing_count': 30},
            'RECUIMA': {'valid_count': 950, 'missing_count': 50},
        }
        
        assert 'ML Model' in report.method_validity
        assert report.method_validity['GRACE']['valid_count'] == 970
    
    def test_sync_report_generation(self):
        """Test generating a human-readable report."""
        report = SynchronizationReport(
            original_size=1000,
            synchronized_size=950,
            samples_removed=50,
            removal_rate=0.05,
        )
        
        report.method_validity = {
            'ML Model': {'valid_count': 1000, 'missing_count': 0},
            'GRACE': {'valid_count': 970, 'missing_count': 30},
        }
        
        report_text = report.report()
        
        assert isinstance(report_text, str)
        assert '1000' in report_text
        assert '950' in report_text
        assert 'GRACE' in report_text


class TestFindStemColumn:
    """Tests for _find_stemi_column method."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a mock orchestrator."""
        orch = ClinicalScaleMLOrchestrator(
            model=None,
            test_df=pd.DataFrame({'dummy': [1, 2, 3]}),
        )
        return orch
    
    def test_find_anterior_column(self, orchestrator):
        """Test finding the 'anterior' STEMI indicator column."""
        df = pd.DataFrame({
            'anterior': [1, 0, 1],
            'age': [45, 52, 68],
        })
        
        col = orchestrator._find_stemi_column(df)
        assert col == 'anterior'
    
    def test_find_stemi_type_column(self, orchestrator):
        """Test finding 'stemi_type' column."""
        df = pd.DataFrame({
            'stemi_type': [1, 0, 1],
            'age': [45, 52, 68],
        })
        
        col = orchestrator._find_stemi_column(df)
        assert col == 'stemi_type'
    
    def test_find_type_of_stemi_column(self, orchestrator):
        """Test finding 'type_of_stemi' column."""
        df = pd.DataFrame({
            'type_of_stemi': [1, 0, 1],
            'age': [45, 52, 68],
        })
        
        col = orchestrator._find_stemi_column(df)
        assert col == 'type_of_stemi'
    
    def test_find_st_elevation_column(self, orchestrator):
        """Test finding 'st_elevation' column."""
        df = pd.DataFrame({
            'st_elevation': [True, False, True],
            'age': [45, 52, 68],
        })
        
        col = orchestrator._find_stemi_column(df)
        assert col == 'st_elevation'
    
    def test_no_stemi_column_found(self, orchestrator):
        """Test behavior when no STEMI column is found."""
        df = pd.DataFrame({
            'age': [45, 52, 68],
            'heart_rate': [78, 85, 92],
        })
        
        col = orchestrator._find_stemi_column(df)
        assert col is None
    
    def test_finds_first_matching_column(self, orchestrator):
        """Test that it returns the first matching column in candidate order."""
        df = pd.DataFrame({
            'anterior': [1, 0, 1],
            'stemi_type': [1, 0, 1],  # Both present
            'age': [45, 52, 68],
        })
        
        col = orchestrator._find_stemi_column(df)
        # Should return 'anterior' because it's first in candidate_names
        assert col == 'anterior'


class TestSubgroupComparison:
    """Tests for _run_comparison_for_subgroup method."""
    
    @pytest.fixture
    def orchestrator_with_data(self):
        """Create an orchestrator with test data."""
        test_df = pd.DataFrame({
            'age': [45, 52, 68, 55, 62],
            'heart_rate': [78, 85, 92, 71, 88],
            'systolic_bp': [145, 152, 138, 165, 142],
            'creatinine': [0.9, 1.2, 1.5, 0.8, 1.1],
            'sex': ['M', 'F', 'M', 'M', 'F'],
            'anterior': [1, 1, 0, 1, 0],
            'muerte_inhospitalaria': [0, 1, 1, 0, 1],
        })
        
        orch = ClinicalScaleMLOrchestrator(
            model=None,
            test_df=test_df,
            target_col='muerte_inhospitalaria',
        )
        return orch
    
    def test_subgroup_filtering(self, orchestrator_with_data):
        """Test that subgroup filtering works correctly."""
        stemi_df = orchestrator_with_data.test_df[
            orchestrator_with_data.test_df['anterior'] == 1
        ]
        
        assert len(stemi_df) == 3
        assert all(stemi_df['anterior'] == 1)
    
    def test_valid_mask_intersection(self, orchestrator_with_data):
        """Test that valid mask correctly identifies subgroup samples."""
        # Create a valid mask (simulated)
        valid_mask = np.array([True, True, False, True, False])
        
        stemi_df = orchestrator_with_data.test_df[
            orchestrator_with_data.test_df['anterior'] == 1
        ]
        
        all_indices = np.arange(len(orchestrator_with_data.test_df))
        subgroup_indices = stemi_df.index.values
        mask_in_subgroup = np.isin(all_indices, subgroup_indices)
        
        # Intersection
        subgroup_valid = valid_mask & mask_in_subgroup
        
        # Should only keep valid samples that are in STEMI subgroup
        assert subgroup_valid.sum() == 2  # Indices 0 and 3


class TestOrchestratorSubgroupAnalysis:
    """Tests for compare_all with subgroup analysis."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator with necessary methods."""
        orch = ClinicalScaleMLOrchestrator(
            model=None,
            test_df=pd.DataFrame({
                'age': range(1, 101),
                'heart_rate': np.random.randint(50, 120, 100),
                'systolic_bp': np.random.randint(90, 200, 100),
                'creatinine': np.random.uniform(0.5, 2.0, 100),
                'anterior': np.random.choice([0, 1], 100),
                'muerte_inhospitalaria': np.random.choice([0, 1], 100),
            }),
        )
        
        # Mock synchronize_predictions
        valid_mask = np.ones(100, dtype=bool)
        sync_preds = {
            'ML Model': np.random.uniform(0, 1, 100),
            'GRACE': np.random.uniform(0, 1, 100),
            'RECUIMA': np.random.uniform(0, 1, 100),
            'TIMI-STEMI': np.random.uniform(0, 1, 100),
            'TIMI-NSTEMI': np.random.uniform(0, 1, 100),
        }
        
        orch.synchronize_predictions = Mock(return_value=(valid_mask, sync_preds))
        
        # Mock comparison functions
        orch._run_comparison_for_subgroup = Mock(return_value={
            'grace_vs_ml': {'auc_ml': 0.75, 'auc_scale': 0.72},
            'timi_vs_ml': {'auc_ml': 0.75, 'auc_timi': 0.70},
        })
        
        return orch
    
    def test_compare_all_with_subgroup_analysis(self, mock_orchestrator):
        """Test compare_all with subgroup_analysis=True."""
        results = mock_orchestrator.compare_all(subgroup_analysis=True, skip_timi=False)
        
        assert 'global' in results
        assert 'STEMI' in results or 'NSTEMI' in results
    
    def test_compare_all_returns_nested_structure(self, mock_orchestrator):
        """Test that compare_all returns properly nested results."""
        results = mock_orchestrator.compare_all(subgroup_analysis=True)
        
        # With subgroup analysis, results should have 'global' key
        assert isinstance(results, dict)
        assert 'global' in results
    
    def test_subgroup_comparison_called(self, mock_orchestrator):
        """Test that _run_comparison_for_subgroup is called for subgroups."""
        mock_orchestrator.compare_all(subgroup_analysis=True, skip_timi=False)
        
        # Should be called for STEMI and NSTEMI subgroups
        assert mock_orchestrator._run_comparison_for_subgroup.called


class TestSynchronizeWithReport:
    """Tests for synchronize_predictions with SynchronizationReport generation."""
    
    def test_sync_predictions_creates_report(self):
        """Test that synchronize_predictions creates a SynchronizationReport."""
        orch = ClinicalScaleMLOrchestrator(
            model=None,
            test_df=pd.DataFrame({'dummy': [1, 2, 3]}),
        )
        
        # Manually set predictions
        orch._predictions = {
            'ML Model': np.array([0.1, 0.5, 0.9]),
            'GRACE': np.array([0.2, 0.4, 0.8]),
            'RECUIMA': np.array([np.nan, 0.6, 0.7]),  # Missing value
        }
        
        valid_mask, sync_preds = orch.synchronize_predictions()
        
        # Should create a report
        assert orch._sync_report is not None
        assert isinstance(orch._sync_report, SynchronizationReport)
    
    def test_sync_report_tracks_removal(self):
        """Test that SynchronizationReport tracks sample removal."""
        orch = ClinicalScaleMLOrchestrator(
            model=None,
            test_df=pd.DataFrame({'dummy': [1, 2, 3, 4, 5]}),
        )
        
        orch._predictions = {
            'ML Model': np.array([0.1, 0.5, 0.9, 0.3, 0.7]),
            'GRACE': np.array([0.2, 0.4, 0.8, np.nan, 0.6]),
            'RECUIMA': np.array([0.15, np.nan, 0.75, 0.25, 0.65]),
        }
        
        valid_mask, sync_preds = orch.synchronize_predictions()
        report = orch.get_sync_report()
        
        assert report.original_size == 5
        assert report.samples_removed == 4  # Only sample 0 is fully valid
        assert report.synchronized_size == 1
    
    def test_get_sync_report(self):
        """Test the get_sync_report() method."""
        orch = ClinicalScaleMLOrchestrator(
            model=None,
            test_df=pd.DataFrame({'dummy': [1, 2, 3]}),
        )
        
        # Before synchronize_predictions
        assert orch.get_sync_report() is None
        
        # After synchronize_predictions
        orch._predictions = {
            'ML Model': np.array([0.1, 0.5, 0.9]),
            'GRACE': np.array([0.2, 0.4, 0.8]),
        }
        orch.synchronize_predictions()
        
        report = orch.get_sync_report()
        assert report is not None
        assert isinstance(report, SynchronizationReport)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
