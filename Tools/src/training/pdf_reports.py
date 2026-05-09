"""PDF Reports for Model Training.

This module provides functions to generate comprehensive PDF reports
for model training results, including hyperparameters, CV metrics,
learning curves, and statistical comparisons.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import tempfile

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

from src.reporting import PDFReportGenerator, ReportSection, ReportTemplate
from src.models.metadata import ModelMetadata


class TrainingReport(PDFReportGenerator):
    """Generate PDF report for model training results."""
    
    def __init__(
        self,
        training_results: Dict[str, Any],
        models_metadata: Dict[str, ModelMetadata],
        template: Optional[ReportTemplate] = None
    ):
        """
        Initialize training report.
        
        Args:
            training_results: Dictionary with training results per model.
                Expected keys: 'cv_results', 'statistical_comparison', 
                'learning_curves_paths', 'best_model_name'
            models_metadata: Dictionary mapping model names to their metadata.
            template: Report template configuration.
        """
        if template is None:
            template = ReportTemplate(
                title="Reporte de Entrenamiento",
                subtitle="Modelos de Predicción de Mortalidad AMI",
                author="Sistema Automatizado",
                organization="Mortality AMI Predictor",
                include_toc=True,
            )
        
        super().__init__(template)
        
        self.training_results = training_results
        self.models_metadata = models_metadata
        self.temp_dir = None
    
    def _save_plot_as_image(self, fig: go.Figure, filename: str) -> Path:
        """Save Plotly figure as PNG image."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        
        image_path = Path(self.temp_dir) / filename
        
        try:
            fig.write_image(str(image_path), width=800, height=600)
        except Exception:
            try:
                import kaleido
                fig.write_image(str(image_path), width=800, height=600, engine="kaleido")
            except Exception:
                pass
        
        return image_path
    
    def _create_overview_section(self) -> ReportSection:
        """Create overview section."""
        content = []
        
        content.append(Paragraph(
            f"<b>Fecha del entrenamiento:</b> {datetime.now().strftime('%d de %B de %Y')}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        n_models = len(self.models_metadata)
        content.append(Paragraph(
            f"<b>Modelos entrenados:</b> {n_models}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # List trained models
        content.append(Paragraph(
            "<b>Modelos incluidos en este reporte:</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        for model_name in self.models_metadata.keys():
            content.append(Paragraph(
                f"• {model_name}",
                self.styles['CustomBody']
            ))
        
        content.append(Spacer(1, 0.2 * inch))
        
        content.append(Paragraph(
            "Este reporte presenta los resultados del entrenamiento de modelos predictivos, "
            "incluyendo hiperparámetros optimizados, métricas de validación cruzada, "
            "curvas de aprendizaje y comparaciones estadísticas.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="1. Resumen Ejecutivo",
            level=1,
            content=content
        )
    
    def _create_model_section(self, model_name: str) -> ReportSection:
        """Create detailed section for a single model."""
        content = []
        
        metadata = self.models_metadata.get(model_name)
        if not metadata:
            content.append(Paragraph(
                f"<i>No se encontraron metadatos para {model_name}</i>",
                self.styles['CustomBody']
            ))
            return ReportSection(
                title=f"Modelo: {model_name}",
                level=2,
                content=content
            )
        
        # Model type and description
        content.append(Paragraph(
            f"<b>Tipo:</b> {metadata.model_type}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            f"<b>Tarea:</b> {metadata.task}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Hyperparameters
        if metadata.hyperparameters:
            content.append(Paragraph(
                "<b>Hiperparámetros Optimizados</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            # Create table
            hyperparam_data = [['Hiperparámetro', 'Valor']]
            for key, value in metadata.hyperparameters.items():
                # Format value
                if isinstance(value, float):
                    value_str = f"{value:.6f}"
                else:
                    value_str = str(value)
                hyperparam_data.append([key, value_str])
            
            table = self.create_table(hyperparam_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
        
        # Training configuration
        if metadata.training:
            content.append(Paragraph(
                "<b>Configuración de Entrenamiento</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            train_data = [
                ['Parámetro', 'Valor'],
                ['Estrategia CV', metadata.training.cv_strategy or 'N/A'],
                ['Número de folds', str(metadata.training.n_splits) if metadata.training.n_splits else 'N/A'],
                ['Repeticiones', str(metadata.training.n_repeats) if metadata.training.n_repeats else 'N/A'],
                ['Total de corridas CV', str(metadata.training.total_cv_runs) if metadata.training.total_cv_runs else 'N/A'],
                ['Métrica de scoring', metadata.training.scoring_metric or 'N/A'],
                ['Duración', f"{metadata.training.training_duration_seconds:.2f}s" if metadata.training.training_duration_seconds else 'N/A'],
            ]
            
            table = self.create_table(train_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
        
        # Performance metrics
        if metadata.performance:
            content.append(Paragraph(
                "<b>Métricas de Rendimiento (Validación Cruzada)</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            metric_name = metadata.training.scoring_metric if metadata.training else "Score"
            
            perf_data = [
                ['Métrica', 'Valor'],
                [f'{metric_name} (Media)', f"{metadata.performance.mean_score:.6f}"],
                [f'{metric_name} (Desv. Est.)', f"{metadata.performance.std_score:.6f}"],
                [f'{metric_name} (Mínimo)', f"{metadata.performance.min_score:.6f}"],
                [f'{metric_name} (Máximo)', f"{metadata.performance.max_score:.6f}"],
            ]
            
            table = self.create_table(perf_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
            
            # Distribution of CV scores
            if metadata.performance.all_scores:
                content.append(Paragraph(
                    "<b>Distribución de Scores de Validación Cruzada</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                scores = metadata.performance.all_scores
                dist_data = [['Estadística', 'Valor']]
                dist_data.append(['Número de corridas', str(len(scores))])
                dist_data.append(['Media', f"{np.mean(scores):.6f}"])
                dist_data.append(['Mediana', f"{np.median(scores):.6f}"])
                dist_data.append(['Desv. Est.', f"{np.std(scores):.6f}"])
                dist_data.append(['Mínimo', f"{np.min(scores):.6f}"])
                dist_data.append(['Máximo', f"{np.max(scores):.6f}"])
                dist_data.append(['Q1 (25%)', f"{np.percentile(scores, 25):.6f}"])
                dist_data.append(['Q3 (75%)', f"{np.percentile(scores, 75):.6f}"])
                
                table = self.create_table(dist_data, has_header=True)
                content.append(table)
                content.append(Spacer(1, 0.2 * inch))
        
        # Dataset information
        if metadata.dataset:
            content.append(Paragraph(
                "<b>Información del Dataset</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            dataset_data = [
                ['Información', 'Valor'],
                ['Muestras de entrenamiento', f"{metadata.dataset.train_samples:,}"],
                ['Muestras de prueba', f"{metadata.dataset.test_samples:,}"],
                ['Número de features', f"{metadata.dataset.n_features}"],
                ['Target', metadata.dataset.target_column or 'N/A'],
            ]
            
            # Add class distribution if available
            if metadata.dataset.class_distribution_train:
                for cls, prop in metadata.dataset.class_distribution_train.items():
                    dataset_data.append([f'Clase {cls} (train)', f"{prop*100:.2f}%"])
            
            table = self.create_table(dataset_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
        
        # Learning curve
        lc_paths = self.training_results.get('learning_curves_paths', {})
        if model_name in lc_paths:
            lc_path = Path(lc_paths[model_name])
            
            if lc_path.exists():
                content.append(Paragraph(
                    "<b>Curva de Aprendizaje</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                content.append(Paragraph(
                    "La curva de aprendizaje muestra cómo varía el rendimiento del modelo "
                    "según el tamaño del conjunto de entrenamiento.",
                    self.styles['CustomBody']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                try:
                    img_elements = self.create_image(
                        lc_path,
                        width=5.5 * inch,
                        caption=f"Curva de Aprendizaje: {model_name}"
                    )
                    content.extend(img_elements)
                except Exception as e:
                    content.append(Paragraph(
                        f"<i>Error al cargar curva de aprendizaje: {str(e)}</i>",
                        self.styles['CustomBody']
                    ))
        
        return ReportSection(
            title=f"Modelo: {model_name}",
            level=2,
            content=content,
            page_break_after=True
        )
    
    def _create_comparison_section(self) -> ReportSection:
        """Create statistical comparison section."""
        content = []
        
        statistical_comp = self.training_results.get('statistical_comparison')
        
        if not statistical_comp:
            content.append(Paragraph(
                "<i>No hay información de comparación estadística disponible.</i>",
                self.styles['CustomBody']
            ))
            return ReportSection(
                title="4. Comparación Estadística",
                level=1,
                content=content
            )
        
        content.append(Paragraph(
            "Se realizaron tests estadísticos para comparar el rendimiento de los modelos "
            "y determinar si existen diferencias significativas.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Normality tests
        normality_tests = statistical_comp.get('normality_tests', {})
        if normality_tests:
            content.append(Paragraph(
                "<b>Tests de Normalidad (Shapiro-Wilk)</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            norm_data = [['Modelo', 'Estadístico W', 'p-valor', 'Normal (α=0.05)']]
            for model_name, test_result in normality_tests.items():
                is_normal = "Sí" if test_result['p_value'] > 0.05 else "No"
                norm_data.append([
                    model_name,
                    f"{test_result['statistic']:.6f}",
                    f"{test_result['p_value']:.6f}",
                    is_normal
                ])
            
            table = self.create_table(norm_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
        
        # Pairwise comparisons
        pairwise_tests = statistical_comp.get('pairwise_tests', {})
        if pairwise_tests:
            content.append(Paragraph(
                "<b>Comparaciones por Pares</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            content.append(Paragraph(
                "Se utilizó t-test para distribuciones normales y Mann-Whitney U para distribuciones no normales.",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            pair_data = [['Modelo 1', 'Modelo 2', 'Test', 'Estadístico', 'p-valor', 'Significativo']]
            
            for pair_key, test_result in pairwise_tests.items():
                parts = pair_key.split('_vs_')
                if len(parts) == 2:
                    model1, model2 = parts
                    is_sig = "Sí" if test_result['p_value'] < 0.05 else "No"
                    pair_data.append([
                        model1,
                        model2,
                        test_result['test_type'],
                        f"{test_result['statistic']:.6f}",
                        f"{test_result['p_value']:.6f}",
                        is_sig
                    ])
            
            table = self.create_table(pair_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
        
        # Interpretation
        content.append(Paragraph(
            "<b>Interpretación</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            "• Un p-valor < 0.05 indica diferencias estadísticamente significativas entre modelos.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• Si no hay diferencias significativas, se recomienda el modelo más simple.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• Considerar también métricas de negocio y tiempo de inferencia.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="4. Comparación Estadística",
            level=1,
            content=content,
            page_break_before=True
        )
    
    def _create_best_model_section(self) -> ReportSection:
        """Create best model selection section."""
        content = []
        
        best_name = self.training_results.get('best_model_name')
        
        if not best_name:
            content.append(Paragraph(
                "<i>No se seleccionó un mejor modelo.</i>",
                self.styles['CustomBody']
            ))
            return ReportSection(
                title="5. Modelo Seleccionado",
                level=1,
                content=content
            )
        
        content.append(Paragraph(
            f"<b>Mejor Modelo:</b> {best_name}",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        best_metadata = self.models_metadata.get(best_name)
        
        if best_metadata and best_metadata.performance:
            metric_name = best_metadata.training.scoring_metric if best_metadata.training else "Score"
            
            content.append(Paragraph(
                f"Este modelo fue seleccionado por tener el mejor rendimiento en validación cruzada "
                f"({metric_name} = {best_metadata.performance.mean_score:.6f} ± {best_metadata.performance.std_score:.6f}).",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.2 * inch))
            
            # Summary table
            summary_data = [
                ['Métrica', 'Valor'],
                ['Modelo', best_name],
                ['Tipo', best_metadata.model_type],
                [f'{metric_name} (Media)', f"{best_metadata.performance.mean_score:.6f}"],
                [f'{metric_name} (Desv. Est.)', f"{best_metadata.performance.std_score:.6f}"],
            ]
            
            table = self.create_table(summary_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
        
        content.append(Paragraph(
            "<b>Próximos Pasos:</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            "• Evaluar el modelo en el conjunto de prueba.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• Analizar la importancia de features y explicabilidad.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• Considerar calibración de probabilidades si es necesario.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• Validar con datos externos si están disponibles.",
            self.styles['CustomBody']
        ))
        
        return ReportSection(
            title="5. Modelo Seleccionado",
            level=1,
            content=content
        )
    
    def _build_content(self) -> List[ReportSection]:
        """Build the report content."""
        sections = []
        
        # Overview
        sections.append(self._create_overview_section())
        
        # Performance summary table
        summary_content = []
        
        summary_content.append(Paragraph(
            "La siguiente tabla resume las métricas de rendimiento de todos los modelos entrenados:",
            self.styles['CustomBody']
        ))
        summary_content.append(Spacer(1, 0.2 * inch))
        
        # Create summary table
        summary_data = [['Modelo', 'Tipo', 'Score (Media)', 'Score (Std)', 'Duración (s)']]
        
        for model_name, metadata in self.models_metadata.items():
            if metadata.performance:
                summary_data.append([
                    model_name,
                    metadata.model_type,
                    f"{metadata.performance.mean_score:.6f}",
                    f"{metadata.performance.std_score:.6f}",
                    f"{metadata.training.training_duration_seconds:.2f}" if metadata.training else 'N/A'
                ])
        
        table = self.create_table(summary_data, has_header=True)
        summary_content.append(table)
        summary_content.append(Spacer(1, 0.2 * inch))
        
        sections.append(ReportSection(
            title="2. Resumen de Rendimiento",
            level=1,
            content=summary_content
        ))
        
        # Detailed sections for each model
        detail_intro = [
            Paragraph(
                "Las siguientes secciones presentan el análisis detallado de cada modelo entrenado, "
                "incluyendo hiperparámetros, configuración de entrenamiento, métricas de validación "
                "cruzada y curvas de aprendizaje.",
                self.styles['CustomBody']
            ),
            Spacer(1, 0.2 * inch),
        ]
        
        sections.append(ReportSection(
            title="3. Análisis Detallado por Modelo",
            level=1,
            content=detail_intro,
            page_break_before=True
        ))
        
        for model_name in self.models_metadata.keys():
            sections.append(self._create_model_section(model_name))
        
        # Statistical comparison
        sections.append(self._create_comparison_section())
        
        # Best model
        sections.append(self._create_best_model_section())
        
        return sections


def generate_training_pdf(
    training_results: Dict[str, Any],
    models_metadata: Dict[str, ModelMetadata],
    output_path: str | Path,
    template: Optional[ReportTemplate] = None,
    progress_callback: Optional[callable] = None
) -> Path:
    """
    Generate training PDF report.
    
    Args:
        training_results: Dictionary with training results.
            Expected keys: 'cv_results', 'statistical_comparison',
            'learning_curves_paths', 'best_model_name'
        models_metadata: Dictionary mapping model names to their ModelMetadata.
        output_path: Path where to save the PDF.
        template: Optional report template configuration.
        progress_callback: Optional callback(progress: float, message: str).
    
    Returns:
        Path to generated PDF.
    
    Example:
        ```python
        from src.training.pdf_reports import generate_training_pdf
        from src.data_load import load_model_with_metadata
        
        # Load models with metadata
        models_metadata = {}
        for model_name in ['dtree', 'knn', 'xgb']:
            model_path = get_latest_model(model_name)
            _, metadata = load_model_with_metadata(model_path)
            models_metadata[model_name] = metadata
        
        # Training results from trainer
        training_results = {
            'best_model_name': 'xgb',
            'statistical_comparison': {...},
            'learning_curves_paths': {...}
        }
        
        pdf_path = generate_training_pdf(
            training_results=training_results,
            models_metadata=models_metadata,
            output_path="reports/training_report.pdf"
        )
        ```
    """
    report = TrainingReport(
        training_results=training_results,
        models_metadata=models_metadata,
        template=template
    )
    
    return report.generate(
        output_path=output_path,
        async_mode=False,
        progress_callback=progress_callback
    )
