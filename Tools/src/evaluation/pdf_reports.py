"""PDF Reports for Model Evaluation.

This module provides functions to generate comprehensive PDF reports
for model evaluation, including metrics, ROC curves, confusion matrices,
calibration plots, and decision curve analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import tempfile

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

from src.reporting import PDFReportGenerator, ReportSection, ReportTemplate
from src.models.metadata import ModelMetadata


class EvaluationReport(PDFReportGenerator):
    """Generate PDF report for model evaluation results."""
    
    def __init__(
        self,
        models_data: Dict[str, Dict[str, Any]],
        template: Optional[ReportTemplate] = None
    ):
        """
        Initialize evaluation report.
        
        Args:
            models_data: Dictionary mapping model names to their evaluation data.
                Each model dict should contain:
                - 'metadata': ModelMetadata object
                - 'y_true': True labels
                - 'y_pred': Predicted labels
                - 'y_proba': Predicted probabilities
                - 'metrics': Dict with computed metrics
                - 'plots': Dict with paths to saved plots
            template: Report template configuration.
        """
        if template is None:
            template = ReportTemplate(
                title="Reporte de Evaluación",
                subtitle="Evaluación de Modelos en Conjunto de Prueba",
                author="Sistema Automatizado",
                organization="Mortality AMI Predictor",
                include_toc=True,
            )
        
        super().__init__(template)
        
        self.models_data = models_data
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
            f"<b>Fecha de evaluación:</b> {datetime.now().strftime('%d de %B de %Y')}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        n_models = len(self.models_data)
        content.append(Paragraph(
            f"<b>Modelos evaluados:</b> {n_models}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # List evaluated models
        content.append(Paragraph(
            "<b>Modelos incluidos en este reporte:</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        for model_name in self.models_data.keys():
            content.append(Paragraph(
                f"• {model_name}",
                self.styles['CustomBody']
            ))
        
        content.append(Spacer(1, 0.2 * inch))
        
        content.append(Paragraph(
            "Este reporte presenta los resultados de la evaluación de modelos en el conjunto "
            "de prueba, incluyendo métricas de clasificación, curvas ROC y PR, matrices de "
            "confusión, calibración de probabilidades y análisis de curvas de decisión.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="1. Resumen Ejecutivo",
            level=1,
            content=content
        )
    
    def _create_comparison_section(self) -> ReportSection:
        """Create models comparison section."""
        content = []
        
        content.append(Paragraph(
            "La siguiente tabla compara las métricas de rendimiento de todos los modelos "
            "evaluados en el conjunto de prueba:",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Create comparison table
        table_data = [['Modelo', 'AUROC', 'Precisión', 'Recall', 'F1-Score', 'Accuracy']]
        
        for model_name, data in self.models_data.items():
            metrics = data.get('metrics', {})
            table_data.append([
                model_name,
                f"{metrics.get('auroc', 0):.4f}" if 'auroc' in metrics else 'N/A',
                f"{metrics.get('precision', 0):.4f}" if 'precision' in metrics else 'N/A',
                f"{metrics.get('recall', 0):.4f}" if 'recall' in metrics else 'N/A',
                f"{metrics.get('f1', 0):.4f}" if 'f1' in metrics else 'N/A',
                f"{metrics.get('accuracy', 0):.4f}" if 'accuracy' in metrics else 'N/A',
            ])
        
        table = self.create_table(table_data, has_header=True)
        content.append(table)
        content.append(Spacer(1, 0.2 * inch))
        
        # Find best model by AUROC
        best_model = max(
            self.models_data.items(),
            key=lambda x: x[1].get('metrics', {}).get('auroc', 0)
        )
        
        content.append(Paragraph(
            f"<b>Mejor modelo por AUROC:</b> {best_model[0]} "
            f"(AUROC = {best_model[1].get('metrics', {}).get('auroc', 0):.4f})",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="2. Comparación de Modelos",
            level=1,
            content=content
        )
    
    def _create_model_section(self, model_name: str) -> ReportSection:
        """Create detailed section for a single model."""
        content = []
        
        data = self.models_data.get(model_name, {})
        metrics = data.get('metrics', {})
        plots = data.get('plots', {})
        metadata = data.get('metadata')
        
        # Model info
        if metadata:
            content.append(Paragraph(
                f"<b>Tipo:</b> {metadata.model_type}",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.1 * inch))
        
        # Metrics table
        content.append(Paragraph(
            "<b>Métricas de Clasificación</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        metrics_data = [['Métrica', 'Valor']]
        
        metric_names = {
            'auroc': 'AUROC',
            'accuracy': 'Accuracy',
            'precision': 'Precisión',
            'recall': 'Recall (Sensibilidad)',
            'specificity': 'Especificidad',
            'f1': 'F1-Score',
            'auprc': 'AUPRC',
            'brier': 'Brier Score',
        }
        
        for key, name in metric_names.items():
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    metrics_data.append([name, f"{value:.4f}"])
                else:
                    metrics_data.append([name, str(value)])
        
        table = self.create_table(metrics_data, has_header=True)
        content.append(table)
        content.append(Spacer(1, 0.2 * inch))
        
        # Confusion matrix
        if 'y_true' in data and 'y_pred' in data:
            content.append(Paragraph(
                "<b>Matriz de Confusión</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            cm = confusion_matrix(data['y_true'], data['y_pred'])
            
            cm_data = [['', 'Pred: Negativo', 'Pred: Positivo']]
            cm_data.append(['Real: Negativo', str(cm[0, 0]), str(cm[0, 1])])
            cm_data.append(['Real: Positivo', str(cm[1, 0]), str(cm[1, 1])])
            
            table = self.create_table(cm_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.1 * inch))
            
            # Interpretation
            tn, fp, fn, tp = cm.ravel()
            content.append(Paragraph(
                f"• Verdaderos Negativos (TN): {tn}",
                self.styles['CustomBody']
            ))
            content.append(Paragraph(
                f"• Falsos Positivos (FP): {fp}",
                self.styles['CustomBody']
            ))
            content.append(Paragraph(
                f"• Falsos Negativos (FN): {fn}",
                self.styles['CustomBody']
            ))
            content.append(Paragraph(
                f"• Verdaderos Positivos (TP): {tp}",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.2 * inch))
        
        # ROC Curve
        if 'roc_curve' in plots:
            plot_path = Path(plots['roc_curve'])
            if plot_path.exists():
                content.append(Paragraph(
                    "<b>Curva ROC</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                content.append(Paragraph(
                    "La curva ROC muestra el trade-off entre sensibilidad (recall) y especificidad.",
                    self.styles['CustomBody']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                try:
                    img_elements = self.create_image(
                        plot_path,
                        width=5 * inch,
                        caption=f"Curva ROC: {model_name}"
                    )
                    content.extend(img_elements)
                except Exception as e:
                    content.append(Paragraph(
                        f"<i>Error al cargar curva ROC: {str(e)}</i>",
                        self.styles['CustomBody']
                    ))
        
        # PR Curve
        if 'pr_curve' in plots:
            plot_path = Path(plots['pr_curve'])
            if plot_path.exists():
                content.append(Paragraph(
                    "<b>Curva Precisión-Recall</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                content.append(Paragraph(
                    "La curva PR muestra el trade-off entre precisión y recall. "
                    "Es especialmente útil para datasets desbalanceados.",
                    self.styles['CustomBody']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                try:
                    img_elements = self.create_image(
                        plot_path,
                        width=5 * inch,
                        caption=f"Curva Precisión-Recall: {model_name}"
                    )
                    content.extend(img_elements)
                except Exception as e:
                    content.append(Paragraph(
                        f"<i>Error al cargar curva PR: {str(e)}</i>",
                        self.styles['CustomBody']
                    ))
        
        # Calibration curve
        if 'calibration_curve' in plots:
            plot_path = Path(plots['calibration_curve'])
            if plot_path.exists():
                content.append(Paragraph(
                    "<b>Curva de Calibración</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                content.append(Paragraph(
                    "La curva de calibración muestra qué tan bien calibradas están las "
                    "probabilidades predichas. Una curva cercana a la diagonal indica buena calibración.",
                    self.styles['CustomBody']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                try:
                    img_elements = self.create_image(
                        plot_path,
                        width=5 * inch,
                        caption=f"Curva de Calibración: {model_name}"
                    )
                    content.extend(img_elements)
                except Exception as e:
                    content.append(Paragraph(
                        f"<i>Error al cargar curva de calibración: {str(e)}</i>",
                        self.styles['CustomBody']
                    ))
        
        # Decision curve
        if 'decision_curve' in plots:
            plot_path = Path(plots['decision_curve'])
            if plot_path.exists():
                content.append(Paragraph(
                    "<b>Curva de Decisión</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                content.append(Paragraph(
                    "El Decision Curve Analysis evalúa el beneficio clínico neto del modelo "
                    "a diferentes umbrales de probabilidad.",
                    self.styles['CustomBody']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                try:
                    img_elements = self.create_image(
                        plot_path,
                        width=5 * inch,
                        caption=f"Decision Curve Analysis: {model_name}"
                    )
                    content.extend(img_elements)
                except Exception as e:
                    content.append(Paragraph(
                        f"<i>Error al cargar decision curve: {str(e)}</i>",
                        self.styles['CustomBody']
                    ))
        
        return ReportSection(
            title=f"Modelo: {model_name}",
            level=2,
            content=content,
            page_break_after=True
        )
    
    def _build_content(self) -> List[ReportSection]:
        """Build the report content."""
        sections = []
        
        # Overview
        sections.append(self._create_overview_section())
        
        # Comparison
        sections.append(self._create_comparison_section())
        
        # Detailed sections for each model
        detail_intro = [
            Paragraph(
                "Las siguientes secciones presentan el análisis detallado de cada modelo, "
                "incluyendo métricas completas, matriz de confusión, curvas ROC y PR, "
                "calibración y análisis de curvas de decisión.",
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
        
        for model_name in self.models_data.keys():
            sections.append(self._create_model_section(model_name))
        
        # Conclusions
        conclusion_content = [
            Paragraph(
                "Este reporte ha presentado una evaluación exhaustiva de los modelos predictivos "
                "en el conjunto de prueba.",
                self.styles['CustomBody']
            ),
            Spacer(1, 0.2 * inch),
            Paragraph(
                "<b>Recomendaciones:</b>",
                self.styles['CustomHeading3']
            ),
            Spacer(1, 0.1 * inch),
            Paragraph(
                "• Seleccionar el modelo con mejor AUROC y métricas balanceadas.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Considerar el contexto clínico al elegir el umbral de decisión.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Verificar la calibración de probabilidades antes del despliegue.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Realizar análisis de explicabilidad para interpretación clínica.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Validar con datos externos si están disponibles.",
                self.styles['CustomBody']
            ),
        ]
        
        sections.append(ReportSection(
            title="4. Conclusiones y Recomendaciones",
            level=1,
            content=conclusion_content,
            page_break_before=True
        ))
        
        return sections


def generate_evaluation_pdf(
    models_data: Dict[str, Dict[str, Any]],
    output_path: str | Path,
    template: Optional[ReportTemplate] = None,
    progress_callback: Optional[callable] = None
) -> Path:
    """
    Generate evaluation PDF report.
    
    Args:
        models_data: Dictionary mapping model names to their evaluation data.
            Each model dict should contain:
            - 'metadata': ModelMetadata object (optional)
            - 'y_true': True labels
            - 'y_pred': Predicted labels
            - 'y_proba': Predicted probabilities (optional)
            - 'metrics': Dict with computed metrics
            - 'plots': Dict with paths to saved plots (optional)
        output_path: Path where to save the PDF.
        template: Optional report template configuration.
        progress_callback: Optional callback(progress: float, message: str).
    
    Returns:
        Path to generated PDF.
    
    Example:
        ```python
        from src.evaluation.pdf_reports import generate_evaluation_pdf
        from src.data_load import load_model_with_metadata
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        # Prepare data
        models_data = {}
        for model_name in ['dtree', 'knn', 'xgb']:
            model_path = get_latest_model(model_name)
            model, metadata = load_model_with_metadata(model_path)
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Compute metrics
            metrics = {
                'auroc': roc_auc_score(y_test, y_proba),
                'accuracy': accuracy_score(y_test, y_pred),
                # ... other metrics
            }
            
            models_data[model_name] = {
                'metadata': metadata,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'metrics': metrics,
                'plots': {
                    'roc_curve': 'path/to/roc.png',
                    'pr_curve': 'path/to/pr.png',
                }
            }
        
        pdf_path = generate_evaluation_pdf(
            models_data=models_data,
            output_path="reports/evaluation_report.pdf"
        )
        ```
    """
    report = EvaluationReport(
        models_data=models_data,
        template=template
    )
    
    return report.generate(
        output_path=output_path,
        async_mode=False,
        progress_callback=progress_callback
    )
