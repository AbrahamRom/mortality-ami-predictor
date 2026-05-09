"""PDF Reports for Model Explainability.

This module provides functions to generate comprehensive PDF reports
for model explainability, including SHAP analysis, permutation importance,
partial dependence plots, and feature importance visualizations.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import tempfile

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch

from src.reporting import PDFReportGenerator, ReportSection, ReportTemplate
from src.models.metadata import ModelMetadata


class ExplainabilityReport(PDFReportGenerator):
    """Generate PDF report for model explainability analysis."""
    
    def __init__(
        self,
        model_name: str,
        explainability_data: Dict[str, Any],
        metadata: Optional[ModelMetadata] = None,
        template: Optional[ReportTemplate] = None
    ):
        """
        Initialize explainability report.
        
        Args:
            model_name: Name of the model being explained.
            explainability_data: Dictionary containing explainability results:
                - 'feature_importance': Dict with different importance methods
                - 'shap_values': SHAP values array (optional)
                - 'permutation_importance': Permutation importance results
                - 'partial_dependence': PDP data for top features
                - 'plots': Dict with paths to saved plots
                - 'feature_names': List of feature names
                - 'top_features': List of most important features
            metadata: Optional ModelMetadata object.
            template: Report template configuration.
        """
        if template is None:
            template = ReportTemplate(
                title="Reporte de Explicabilidad",
                subtitle=f"Análisis de Interpretabilidad: {model_name}",
                author="Sistema Automatizado",
                organization="Mortality AMI Predictor",
                include_toc=True,
            )
        
        super().__init__(template)
        
        self.model_name = model_name
        self.explainability_data = explainability_data
        self.metadata = metadata
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
            f"<b>Modelo:</b> {self.model_name}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        if self.metadata:
            content.append(Paragraph(
                f"<b>Tipo de modelo:</b> {self.metadata.model_type}",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            f"<b>Fecha de análisis:</b> {datetime.now().strftime('%d de %B de %Y')}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        content.append(Paragraph(
            "Este reporte presenta un análisis exhaustivo de la interpretabilidad del modelo, "
            "incluyendo múltiples métodos de explicación:",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            "• <b>Feature Importance:</b> Importancia global de características según el modelo.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• <b>SHAP Analysis:</b> Valores SHAP para explicaciones locales y globales.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• <b>Permutation Importance:</b> Importancia basada en degradación de rendimiento.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• <b>Partial Dependence:</b> Efecto marginal de características en las predicciones.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Top features summary
        top_features = self.explainability_data.get('top_features', [])
        if top_features:
            content.append(Paragraph(
                "<b>Variables más importantes:</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            for i, feature in enumerate(top_features[:10], 1):
                content.append(Paragraph(
                    f"{i}. {feature}",
                    self.styles['CustomBody']
                ))
        
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="1. Resumen Ejecutivo",
            level=1,
            content=content
        )
    
    def _create_feature_importance_section(self) -> ReportSection:
        """Create feature importance section."""
        content = []
        
        feature_importance = self.explainability_data.get('feature_importance', {})
        plots = self.explainability_data.get('plots', {})
        
        content.append(Paragraph(
            "La importancia de características indica qué variables contribuyen más a las "
            "predicciones del modelo. Se presentan diferentes métodos de cálculo:",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Built-in feature importance
        if 'builtin' in feature_importance:
            content.append(Paragraph(
                "<b>Importancia Nativa del Modelo</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            importance_df = feature_importance['builtin']
            if isinstance(importance_df, pd.DataFrame):
                # Top 15 features table
                top_n = min(15, len(importance_df))
                table_data = [['Característica', 'Importancia']]
                
                for idx in range(top_n):
                    row = importance_df.iloc[idx]
                    feature_name = row.get('feature', row.name)
                    importance_val = row.get('importance', row.values[0])
                    table_data.append([
                        str(feature_name),
                        f"{importance_val:.4f}"
                    ])
                
                table = self.create_table(table_data, has_header=True)
                content.append(table)
                content.append(Spacer(1, 0.2 * inch))
        
        # Feature importance plot
        if 'feature_importance' in plots:
            plot_path = Path(plots['feature_importance'])
            if plot_path.exists():
                content.append(Paragraph(
                    "<b>Gráfico de Importancia</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                try:
                    img_elements = self.create_image(
                        plot_path,
                        width=6 * inch,
                        caption="Importancia de las características más relevantes"
                    )
                    content.extend(img_elements)
                except Exception as e:
                    content.append(Paragraph(
                        f"<i>Error al cargar gráfico: {str(e)}</i>",
                        self.styles['CustomBody']
                    ))
        
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="2. Importancia de Características",
            level=1,
            content=content
        )
    
    def _create_shap_section(self) -> ReportSection:
        """Create SHAP analysis section."""
        content = []
        
        plots = self.explainability_data.get('plots', {})
        
        content.append(Paragraph(
            "SHAP (SHapley Additive exPlanations) es un método basado en teoría de juegos "
            "que proporciona valores de importancia consistentes y localmente precisos para "
            "cada característica.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # SHAP summary plot
        if 'shap_summary' in plots:
            plot_path = Path(plots['shap_summary'])
            if plot_path.exists():
                content.append(Paragraph(
                    "<b>Resumen SHAP</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                content.append(Paragraph(
                    "El gráfico de resumen SHAP muestra la distribución de valores SHAP para "
                    "cada característica. Los puntos representan muestras individuales:",
                    self.styles['CustomBody']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                content.append(Paragraph(
                    "• Posición horizontal: Impacto en la predicción (positivo o negativo).",
                    self.styles['CustomBody']
                ))
                content.append(Paragraph(
                    "• Color: Valor de la característica (rojo = alto, azul = bajo).",
                    self.styles['CustomBody']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                try:
                    img_elements = self.create_image(
                        plot_path,
                        width=6 * inch,
                        caption="Distribución de valores SHAP por característica"
                    )
                    content.extend(img_elements)
                except Exception as e:
                    content.append(Paragraph(
                        f"<i>Error al cargar gráfico SHAP: {str(e)}</i>",
                        self.styles['CustomBody']
                    ))
        
        # SHAP bar plot (mean importance)
        if 'shap_bar' in plots:
            plot_path = Path(plots['shap_bar'])
            if plot_path.exists():
                content.append(Paragraph(
                    "<b>Importancia SHAP Media</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                content.append(Paragraph(
                    "Este gráfico muestra la importancia promedio de cada característica "
                    "basada en la magnitud absoluta de los valores SHAP:",
                    self.styles['CustomBody']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                try:
                    img_elements = self.create_image(
                        plot_path,
                        width=6 * inch,
                        caption="Importancia SHAP promedio (|valor SHAP| medio)"
                    )
                    content.extend(img_elements)
                except Exception as e:
                    content.append(Paragraph(
                        f"<i>Error al cargar gráfico: {str(e)}</i>",
                        self.styles['CustomBody']
                    ))
        
        # SHAP dependence plots
        shap_dep_plots = {k: v for k, v in plots.items() if k.startswith('shap_dependence_')}
        if shap_dep_plots:
            content.append(Paragraph(
                "<b>Gráficos de Dependencia SHAP</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            content.append(Paragraph(
                "Los gráficos de dependencia muestran cómo el valor de una característica "
                "afecta sus valores SHAP (impacto en la predicción):",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.2 * inch))
            
            for plot_key, plot_path in list(shap_dep_plots.items())[:6]:  # Max 6 plots
                plot_path = Path(plot_path)
                if plot_path.exists():
                    feature_name = plot_key.replace('shap_dependence_', '').replace('_', ' ')
                    
                    try:
                        img_elements = self.create_image(
                            plot_path,
                            width=5 * inch,
                            caption=f"Dependencia SHAP: {feature_name}"
                        )
                        content.extend(img_elements)
                    except Exception:
                        pass
        
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="3. Análisis SHAP",
            level=1,
            content=content,
            page_break_before=True
        )
    
    def _create_permutation_section(self) -> ReportSection:
        """Create permutation importance section."""
        content = []
        
        perm_importance = self.explainability_data.get('permutation_importance', {})
        plots = self.explainability_data.get('plots', {})
        
        content.append(Paragraph(
            "La importancia por permutación mide la degradación en el rendimiento del modelo "
            "cuando se permutan aleatoriamente los valores de una característica. Un valor alto "
            "indica que la característica es importante para las predicciones.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Permutation importance table
        if 'results' in perm_importance:
            results = perm_importance['results']
            if isinstance(results, pd.DataFrame):
                content.append(Paragraph(
                    "<b>Resultados de Permutación</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                top_n = min(15, len(results))
                table_data = [['Característica', 'Importancia Media', 'Desv. Estándar']]
                
                for idx in range(top_n):
                    row = results.iloc[idx]
                    table_data.append([
                        str(row.get('feature', row.name)),
                        f"{row.get('importance_mean', 0):.4f}",
                        f"{row.get('importance_std', 0):.4f}"
                    ])
                
                table = self.create_table(table_data, has_header=True)
                content.append(table)
                content.append(Spacer(1, 0.2 * inch))
        
        # Permutation importance plot
        if 'permutation_importance' in plots:
            plot_path = Path(plots['permutation_importance'])
            if plot_path.exists():
                content.append(Paragraph(
                    "<b>Gráfico de Importancia por Permutación</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                try:
                    img_elements = self.create_image(
                        plot_path,
                        width=6 * inch,
                        caption="Importancia por permutación con intervalos de confianza"
                    )
                    content.extend(img_elements)
                except Exception as e:
                    content.append(Paragraph(
                        f"<i>Error al cargar gráfico: {str(e)}</i>",
                        self.styles['CustomBody']
                    ))
        
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="4. Importancia por Permutación",
            level=1,
            content=content
        )
    
    def _create_pdp_section(self) -> ReportSection:
        """Create partial dependence plots section."""
        content = []
        
        plots = self.explainability_data.get('plots', {})
        
        content.append(Paragraph(
            "Los Partial Dependence Plots (PDP) muestran el efecto marginal de una o más "
            "características en las predicciones del modelo, manteniendo las demás características "
            "constantes (promediadas).",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Get all PDP plots
        pdp_plots = {k: v for k, v in plots.items() if k.startswith('pdp_')}
        
        if pdp_plots:
            content.append(Paragraph(
                "<b>Dependencia Parcial de Variables Clave</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            content.append(Paragraph(
                "Los siguientes gráficos muestran cómo cambia la predicción promedio al variar "
                "el valor de cada característica:",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.2 * inch))
            
            for plot_key, plot_path in list(pdp_plots.items())[:8]:  # Max 8 plots
                plot_path = Path(plot_path)
                if plot_path.exists():
                    feature_name = plot_key.replace('pdp_', '').replace('_', ' ')
                    
                    try:
                        img_elements = self.create_image(
                            plot_path,
                            width=5 * inch,
                            caption=f"Dependencia parcial: {feature_name}"
                        )
                        content.extend(img_elements)
                        content.append(Spacer(1, 0.1 * inch))
                    except Exception:
                        pass
        else:
            content.append(Paragraph(
                "<i>No hay gráficos de dependencia parcial disponibles.</i>",
                self.styles['CustomBody']
            ))
        
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="5. Dependencia Parcial (PDP)",
            level=1,
            content=content,
            page_break_before=True
        )
    
    def _create_clinical_interpretation_section(self) -> ReportSection:
        """Create clinical interpretation section."""
        content = []
        
        top_features = self.explainability_data.get('top_features', [])
        
        content.append(Paragraph(
            "La interpretabilidad del modelo es crucial para la adopción clínica. Los resultados "
            "del análisis de explicabilidad permiten:",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            "• <b>Validar coherencia clínica:</b> Verificar que las características importantes "
            "coincidan con el conocimiento médico establecido.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• <b>Identificar sesgos:</b> Detectar si el modelo depende de variables no clínicamente "
            "relevantes o potencialmente sesgadas.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• <b>Facilitar decisiones:</b> Proporcionar explicaciones comprensibles que apoyen la "
            "toma de decisiones clínicas.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• <b>Cumplir normativas:</b> Satisfacer requisitos de transparencia en sistemas de IA "
            "médica (e.g., MDR, GDPR).",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        if top_features:
            content.append(Paragraph(
                "<b>Resumen de Variables Clave</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            content.append(Paragraph(
                f"El modelo identifica {len(top_features)} variables principales que impulsan "
                "sus predicciones. Las más importantes son:",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            for i, feature in enumerate(top_features[:10], 1):
                content.append(Paragraph(
                    f"{i}. <b>{feature}</b>",
                    self.styles['CustomBody']
                ))
            
            content.append(Spacer(1, 0.2 * inch))
        
        content.append(Paragraph(
            "<b>Recomendaciones para Implementación Clínica</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            "• Revisar las variables principales con expertos clínicos para validar su relevancia.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• Considerar el contexto clínico al interpretar predicciones individuales (SHAP local).",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• Documentar las características en el material de soporte para usuarios finales.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• Monitorear la estabilidad de las importancias en datos nuevos (drift detection).",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• Proporcionar explicaciones específicas para cada predicción en la interfaz de usuario.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="6. Interpretación Clínica",
            level=1,
            content=content,
            page_break_before=True
        )
    
    def _build_content(self) -> List[ReportSection]:
        """Build the report content."""
        sections = []
        
        # Overview
        sections.append(self._create_overview_section())
        
        # Feature importance
        sections.append(self._create_feature_importance_section())
        
        # SHAP analysis
        sections.append(self._create_shap_section())
        
        # Permutation importance
        sections.append(self._create_permutation_section())
        
        # Partial dependence
        sections.append(self._create_pdp_section())
        
        # Clinical interpretation
        sections.append(self._create_clinical_interpretation_section())
        
        # Conclusions
        conclusion_content = [
            Paragraph(
                "Este reporte ha presentado un análisis exhaustivo de la explicabilidad del modelo "
                f"<b>{self.model_name}</b>, utilizando múltiples métodos complementarios.",
                self.styles['CustomBody']
            ),
            Spacer(1, 0.2 * inch),
            Paragraph(
                "La combinación de diferentes técnicas de explicabilidad proporciona una comprensión "
                "robusta de cómo el modelo realiza sus predicciones, facilitando la validación clínica "
                "y el cumplimiento normativo.",
                self.styles['CustomBody']
            ),
            Spacer(1, 0.2 * inch),
            Paragraph(
                "<b>Próximos pasos recomendados:</b>",
                self.styles['CustomHeading3']
            ),
            Spacer(1, 0.1 * inch),
            Paragraph(
                "• Validar las características importantes con el equipo clínico.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Integrar explicaciones SHAP en la interfaz de usuario.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Documentar los hallazgos para reguladores y auditores.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Realizar análisis de casos específicos para validación cualitativa.",
                self.styles['CustomBody']
            ),
        ]
        
        sections.append(ReportSection(
            title="7. Conclusiones",
            level=1,
            content=conclusion_content,
            page_break_before=True
        ))
        
        return sections


def generate_explainability_pdf(
    model_name: str,
    explainability_data: Dict[str, Any],
    output_path: str | Path,
    metadata: Optional[ModelMetadata] = None,
    template: Optional[ReportTemplate] = None,
    progress_callback: Optional[callable] = None
) -> Path:
    """
    Generate explainability PDF report.
    
    Args:
        model_name: Name of the model being explained.
        explainability_data: Dictionary containing explainability results:
            - 'feature_importance': Dict with different importance methods
            - 'shap_values': SHAP values array (optional)
            - 'permutation_importance': Permutation importance results
            - 'partial_dependence': PDP data for top features
            - 'plots': Dict with paths to saved plots
            - 'feature_names': List of feature names
            - 'top_features': List of most important features
        output_path: Path where to save the PDF.
        metadata: Optional ModelMetadata object.
        template: Optional report template configuration.
        progress_callback: Optional callback(progress: float, message: str).
    
    Returns:
        Path to generated PDF.
    
    Example:
        ```python
        from src.explainability.pdf_reports import generate_explainability_pdf
        from src.explainability import calculate_shap_values, calculate_permutation_importance
        from src.data_load import load_model_with_metadata
        
        # Load model
        model_path = get_latest_model('xgb')
        model, metadata = load_model_with_metadata(model_path)
        
        # Calculate explanations
        shap_values = calculate_shap_values(model, X_test)
        perm_importance = calculate_permutation_importance(model, X_test, y_test)
        
        # Prepare data
        explainability_data = {
            'feature_importance': {
                'builtin': model.feature_importances_
            },
            'shap_values': shap_values,
            'permutation_importance': {
                'results': perm_importance
            },
            'plots': {
                'shap_summary': 'path/to/shap_summary.png',
                'permutation_importance': 'path/to/perm_imp.png',
                'pdp_age': 'path/to/pdp_age.png',
            },
            'feature_names': feature_names,
            'top_features': ['age', 'bp', 'cholesterol', ...]
        }
        
        pdf_path = generate_explainability_pdf(
            model_name='XGBoost',
            explainability_data=explainability_data,
            output_path="reports/explainability_report.pdf",
            metadata=metadata
        )
        ```
    """
    report = ExplainabilityReport(
        model_name=model_name,
        explainability_data=explainability_data,
        metadata=metadata,
        template=template
    )
    
    return report.generate(
        output_path=output_path,
        async_mode=False,
        progress_callback=progress_callback
    )
