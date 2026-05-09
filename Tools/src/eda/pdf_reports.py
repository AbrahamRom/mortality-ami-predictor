"""PDF Reports for Exploratory Data Analysis (EDA).

This module provides functions to generate comprehensive PDF reports
for univariate, bivariate, and multivariate EDA.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import io
import tempfile

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from reportlab.platypus import Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.units import inch

from src.reporting import PDFReportGenerator, ReportSection, ReportTemplate
from src.eda.univariate import compute_numeric_stats, compute_categorical_stats
from src.eda.visualizations import plot_distribution


class UnivariateEDAReport(PDFReportGenerator):
    """Generate PDF report for univariate EDA."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
        target_col: Optional[str] = None,
        template: Optional[ReportTemplate] = None
    ):
        """
        Initialize univariate EDA report.
        
        Args:
            df: DataFrame to analyze.
            numerical_cols: List of numerical column names.
            categorical_cols: List of categorical column names.
            target_col: Optional target column name.
            template: Report template configuration.
        """
        # Configure template
        if template is None:
            template = ReportTemplate(
                title="Análisis Exploratorio de Datos",
                subtitle="Análisis Univariado",
                author="Sistema Automatizado",
                organization="Mortality AMI Predictor",
                include_toc=True,
            )
        
        super().__init__(template)
        
        self.df = df
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        
        # Temporary directory for plot images
        self.temp_dir = None
    
    def _save_plot_as_image(self, fig: go.Figure, filename: str) -> Path:
        """Save Plotly figure as PNG image."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        
        image_path = Path(self.temp_dir) / filename
        
        # Save as static image
        try:
            fig.write_image(str(image_path), width=800, height=600)
        except Exception:
            # Fallback: use kaleido if available
            try:
                import kaleido
                fig.write_image(str(image_path), width=800, height=600, engine="kaleido")
            except Exception:
                # If all fails, create a placeholder
                pass
        
        return image_path
    
    def _create_overview_section(self) -> ReportSection:
        """Create overview section."""
        content = []
        
        # Dataset summary
        content.append(Paragraph(
            f"<b>Fecha del análisis:</b> {datetime.now().strftime('%d de %B de %Y')}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            f"<b>Número de registros:</b> {len(self.df):,}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            f"<b>Variables numéricas:</b> {len(self.numerical_cols)}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            f"<b>Variables categóricas:</b> {len(self.categorical_cols)}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        if self.target_col:
            content.append(Paragraph(
                f"<b>Variable objetivo:</b> {self.target_col}",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.1 * inch))
        
        content.append(Spacer(1, 0.2 * inch))
        
        # Summary table
        summary_data = [
            ['Métrica', 'Valor'],
            ['Total de registros', f"{len(self.df):,}"],
            ['Total de variables', f"{len(self.df.columns)}"],
            ['Variables numéricas', f"{len(self.numerical_cols)}"],
            ['Variables categóricas', f"{len(self.categorical_cols)}"],
            ['Valores faltantes (total)', f"{self.df.isna().sum().sum():,}"],
            ['Porcentaje de completitud', f"{(1 - self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100:.2f}%"],
        ]
        
        table = self.create_table(summary_data, has_header=True)
        content.append(table)
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="1. Resumen del Dataset",
            level=1,
            content=content
        )
    
    def _create_numerical_variable_section(self, col: str) -> ReportSection:
        """Create section for a numerical variable."""
        content = []
        
        # Compute statistics
        stats = compute_numeric_stats(self.df, col)
        
        # Descriptive statistics
        content.append(Paragraph(
            "<b>Estadísticas Descriptivas</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        stats_data = [
            ['Estadística', 'Valor'],
            ['Observaciones', f"{stats.count:,}"],
            ['Valores faltantes', f"{stats.missing_count} ({stats.missing_percent:.2f}%)"],
            ['Media', f"{stats.mean:.4f}" if stats.mean else "N/A"],
            ['Mediana', f"{stats.median:.4f}" if stats.median else "N/A"],
            ['Desviación estándar', f"{stats.std:.4f}" if stats.std else "N/A"],
            ['Mínimo', f"{stats.min:.4f}" if stats.min else "N/A"],
            ['Q1 (25%)', f"{stats.q25:.4f}" if stats.q25 else "N/A"],
            ['Q3 (75%)', f"{stats.q75:.4f}" if stats.q75 else "N/A"],
            ['Máximo', f"{stats.max:.4f}" if stats.max else "N/A"],
            ['Asimetría', f"{stats.skewness:.4f}" if stats.skewness else "N/A"],
            ['Curtosis', f"{stats.kurtosis:.4f}" if stats.kurtosis else "N/A"],
        ]
        
        table = self.create_table(stats_data, has_header=True)
        content.append(table)
        content.append(Spacer(1, 0.2 * inch))
        
        # Normality test
        content.append(Paragraph(
            "<b>Test de Normalidad (Shapiro-Wilk)</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        try:
            data_clean = self.df[col].dropna()
            if len(data_clean) > 3 and len(data_clean) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data_clean)
                
                normality_data = [
                    ['Métrica', 'Valor', 'Interpretación'],
                    ['Estadístico W', f"{shapiro_stat:.6f}", ''],
                    ['p-valor', f"{shapiro_p:.6f}", 
                     'Normal' if shapiro_p > 0.05 else 'No normal (α=0.05)'],
                ]
                
                table = self.create_table(normality_data, has_header=True)
                content.append(table)
            else:
                content.append(Paragraph(
                    f"<i>Test no aplicable (n={len(data_clean)}). "
                    "El test de Shapiro-Wilk requiere 3 < n ≤ 5000.</i>",
                    self.styles['CustomBody']
                ))
        except Exception as e:
            content.append(Paragraph(
                f"<i>Error al calcular test de normalidad: {str(e)}</i>",
                self.styles['CustomBody']
            ))
        
        content.append(Spacer(1, 0.2 * inch))
        
        # Outliers detection (IQR method)
        content.append(Paragraph(
            "<b>Detección de Outliers (Método IQR)</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        try:
            data_clean = self.df[col].dropna()
            q1, q3 = data_clean.quantile(0.25), data_clean.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
            n_outliers = len(outliers)
            outlier_percent = (n_outliers / len(data_clean)) * 100
            
            outlier_data = [
                ['Métrica', 'Valor'],
                ['Q1 (25%)', f"{q1:.4f}"],
                ['Q3 (75%)', f"{q3:.4f}"],
                ['IQR', f"{iqr:.4f}"],
                ['Límite inferior', f"{lower_bound:.4f}"],
                ['Límite superior', f"{upper_bound:.4f}"],
                ['Número de outliers', f"{n_outliers} ({outlier_percent:.2f}%)"],
            ]
            
            table = self.create_table(outlier_data, has_header=True)
            content.append(table)
        except Exception as e:
            content.append(Paragraph(
                f"<i>Error al detectar outliers: {str(e)}</i>",
                self.styles['CustomBody']
            ))
        
        content.append(Spacer(1, 0.3 * inch))
        
        # Visualization
        content.append(Paragraph(
            "<b>Visualizaciones</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        try:
            # Create distribution plot
            fig = plot_distribution(self.df, col, is_numeric=True, plot_type='histogram')
            img_path = self._save_plot_as_image(fig, f"dist_{col}.png")
            
            if img_path.exists():
                img_elements = self.create_image(
                    img_path,
                    width=5.5 * inch,
                    caption=f"Figura: Distribución de {col}"
                )
                content.extend(img_elements)
            
            # Create boxplot
            fig_box = plot_distribution(self.df, col, is_numeric=True, plot_type='box')
            img_path_box = self._save_plot_as_image(fig_box, f"box_{col}.png")
            
            if img_path_box.exists():
                img_elements_box = self.create_image(
                    img_path_box,
                    width=5.5 * inch,
                    caption=f"Figura: Boxplot de {col}"
                )
                content.extend(img_elements_box)
                
        except Exception as e:
            content.append(Paragraph(
                f"<i>Error al generar visualizaciones: {str(e)}</i>",
                self.styles['CustomBody']
            ))
        
        return ReportSection(
            title=f"Variable: {col}",
            level=2,
            content=content,
            page_break_after=True
        )
    
    def _create_categorical_variable_section(self, col: str) -> ReportSection:
        """Create section for a categorical variable."""
        content = []
        
        # Compute statistics
        stats = compute_categorical_stats(self.df, col)
        
        # Descriptive statistics
        content.append(Paragraph(
            "<b>Estadísticas Descriptivas</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        stats_data = [
            ['Estadística', 'Valor'],
            ['Observaciones', f"{stats.count:,}"],
            ['Valores faltantes', f"{stats.missing_count} ({stats.missing_percent:.2f}%)"],
            ['Categorías únicas', f"{stats.n_categories}"],
            ['Moda', f"{stats.mode}" if stats.mode else "N/A"],
            ['Frecuencia de moda', f"{stats.mode_frequency:,}" if stats.mode_frequency else "N/A"],
        ]
        
        table = self.create_table(stats_data, has_header=True)
        content.append(table)
        content.append(Spacer(1, 0.2 * inch))
        
        # Frequency table (top 10)
        content.append(Paragraph(
            "<b>Tabla de Frecuencias (Top 10)</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        freq_data = [['Categoría', 'Frecuencia', 'Porcentaje']]
        total = stats.count
        
        sorted_cats = sorted(stats.category_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (cat, count) in enumerate(sorted_cats[:10]):
            percent = (count / total) * 100 if total > 0 else 0
            freq_data.append([str(cat), f"{count:,}", f"{percent:.2f}%"])
        
        if len(sorted_cats) > 10:
            freq_data.append(['(Otras categorías)', '...', '...'])
        
        table = self.create_table(freq_data, has_header=True)
        content.append(table)
        content.append(Spacer(1, 0.3 * inch))
        
        # Visualization
        content.append(Paragraph(
            "<b>Visualización</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        try:
            # Create bar chart
            fig = plot_distribution(self.df, col, is_numeric=False, plot_type='bar')
            img_path = self._save_plot_as_image(fig, f"bar_{col}.png")
            
            if img_path.exists():
                img_elements = self.create_image(
                    img_path,
                    width=5.5 * inch,
                    caption=f"Figura: Distribución de {col}"
                )
                content.extend(img_elements)
        except Exception as e:
            content.append(Paragraph(
                f"<i>Error al generar visualización: {str(e)}</i>",
                self.styles['CustomBody']
            ))
        
        return ReportSection(
            title=f"Variable: {col}",
            level=2,
            content=content,
            page_break_after=True
        )
    
    def _build_content(self) -> List[ReportSection]:
        """Build the report content."""
        sections = []
        
        # Overview section
        sections.append(self._create_overview_section())
        
        # Numerical variables section
        if self.numerical_cols:
            intro_content = [
                Paragraph(
                    f"Esta sección presenta el análisis detallado de las {len(self.numerical_cols)} "
                    "variables numéricas del dataset. Para cada variable se incluyen:",
                    self.styles['CustomBody']
                ),
                Spacer(1, 0.1 * inch),
                Paragraph("• Estadísticas descriptivas completas", self.styles['CustomBody']),
                Paragraph("• Test de normalidad (Shapiro-Wilk)", self.styles['CustomBody']),
                Paragraph("• Detección de outliers (método IQR)", self.styles['CustomBody']),
                Paragraph("• Histograma con media y mediana", self.styles['CustomBody']),
                Paragraph("• Boxplot para identificar outliers", self.styles['CustomBody']),
                Spacer(1, 0.2 * inch),
            ]
            
            sections.append(ReportSection(
                title="2. Variables Numéricas",
                level=1,
                content=intro_content,
                page_break_before=True
            ))
            
            # Add section for each numerical variable
            for col in self.numerical_cols:
                sections.append(self._create_numerical_variable_section(col))
        
        # Categorical variables section
        if self.categorical_cols:
            intro_content = [
                Paragraph(
                    f"Esta sección presenta el análisis detallado de las {len(self.categorical_cols)} "
                    "variables categóricas del dataset. Para cada variable se incluyen:",
                    self.styles['CustomBody']
                ),
                Spacer(1, 0.1 * inch),
                Paragraph("• Estadísticas descriptivas (conteo, categorías, moda)", self.styles['CustomBody']),
                Paragraph("• Tabla de frecuencias (top 10 categorías)", self.styles['CustomBody']),
                Paragraph("• Gráfico de barras de distribución", self.styles['CustomBody']),
                Spacer(1, 0.2 * inch),
            ]
            
            sections.append(ReportSection(
                title="3. Variables Categóricas",
                level=1,
                content=intro_content,
                page_break_before=True
            ))
            
            # Add section for each categorical variable
            for col in self.categorical_cols:
                sections.append(self._create_categorical_variable_section(col))
        
        # Conclusions section
        conclusion_content = [
            Paragraph(
                "Este reporte ha presentado un análisis univariado completo del dataset, "
                "examinando cada variable de forma independiente.",
                self.styles['CustomBody']
            ),
            Spacer(1, 0.2 * inch),
            Paragraph(
                "<b>Resumen de hallazgos:</b>",
                self.styles['CustomHeading3']
            ),
            Spacer(1, 0.1 * inch),
            Paragraph(
                f"• Se analizaron {len(self.numerical_cols)} variables numéricas y "
                f"{len(self.categorical_cols)} variables categóricas.",
                self.styles['CustomBody']
            ),
            Paragraph(
                f"• El dataset contiene {len(self.df):,} registros.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Se identificaron valores faltantes, outliers y se evaluó la normalidad "
                "de las variables numéricas.",
                self.styles['CustomBody']
            ),
            Spacer(1, 0.2 * inch),
            Paragraph(
                "<b>Próximos pasos:</b>",
                self.styles['CustomHeading3']
            ),
            Spacer(1, 0.1 * inch),
            Paragraph(
                "• Realizar análisis bivariado para estudiar relaciones entre variables.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Realizar análisis multivariado (correlaciones, PCA).",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Proceder con la limpieza y preprocesamiento de datos.",
                self.styles['CustomBody']
            ),
        ]
        
        sections.append(ReportSection(
            title="4. Conclusiones",
            level=1,
            content=conclusion_content,
            page_break_before=True
        ))
        
        return sections


def generate_univariate_pdf(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    output_path: str | Path,
    target_col: Optional[str] = None,
    template: Optional[ReportTemplate] = None,
    progress_callback: Optional[callable] = None
) -> Path:
    """
    Generate univariate EDA PDF report.
    
    Args:
        df: DataFrame to analyze.
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        output_path: Path where to save the PDF.
        target_col: Optional target column name.
        template: Optional report template configuration.
        progress_callback: Optional callback(progress: float, message: str).
    
    Returns:
        Path to generated PDF.
    
    Example:
        ```python
        from src.eda.pdf_reports import generate_univariate_pdf
        
        pdf_path = generate_univariate_pdf(
            df=data,
            numerical_cols=['age', 'bp', 'cholesterol'],
            categorical_cols=['gender', 'smoking'],
            output_path="reports/eda_univariate.pdf",
            target_col="mortality"
        )
        ```
    """
    report = UnivariateEDAReport(
        df=df,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        template=template
    )
    
    return report.generate(
        output_path=output_path,
        async_mode=False,
        progress_callback=progress_callback
    )


# ============================================================================
# BIVARIATE ANALYSIS REPORT
# ============================================================================

class BivariateEDAReport(PDFReportGenerator):
    """Generate PDF report for bivariate EDA."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        correlation_threshold: float = 0.7,
        template: Optional[ReportTemplate] = None
    ):
        """
        Initialize bivariate EDA report.
        
        Args:
            df: DataFrame to analyze.
            numerical_cols: List of numerical column names.
            correlation_threshold: Minimum absolute correlation to include.
            template: Report template configuration.
        """
        from src.eda.bivariate import analyze_numeric_numeric
        from src.eda.visualizations import plot_correlation_matrix, plot_scatter
        
        if template is None:
            template = ReportTemplate(
                title="Análisis Exploratorio de Datos",
                subtitle="Análisis Bivariado",
                author="Sistema Automatizado",
                organization="Mortality AMI Predictor",
                include_toc=True,
            )
        
        super().__init__(template)
        
        self.df = df
        self.numerical_cols = numerical_cols
        self.correlation_threshold = correlation_threshold
        self.analyze_numeric_numeric = analyze_numeric_numeric
        self.plot_correlation_matrix = plot_correlation_matrix
        self.plot_scatter = plot_scatter
        
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
            f"<b>Fecha del análisis:</b> {datetime.now().strftime('%d de %B de %Y')}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            f"<b>Variables numéricas analizadas:</b> {len(self.numerical_cols)}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            f"<b>Umbral de correlación:</b> |r| ≥ {self.correlation_threshold}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        content.append(Paragraph(
            "Este reporte analiza las relaciones bivariadas entre variables numéricas, "
            "identificando correlaciones significativas y patrones de asociación.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="1. Resumen",
            level=1,
            content=content
        )
    
    def _create_correlation_matrix_section(self) -> ReportSection:
        """Create correlation matrix section."""
        content = []
        
        # Compute correlation matrix
        df_numeric = self.df[self.numerical_cols].select_dtypes(include=[np.number])
        corr_matrix = df_numeric.corr()
        
        content.append(Paragraph(
            "La matriz de correlación muestra las relaciones lineales entre todas las "
            "variables numéricas. Los valores cercanos a +1 o -1 indican correlaciones fuertes.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Create correlation matrix plot
        try:
            fig = self.plot_correlation_matrix(
                self.df,
                self.numerical_cols,
                method='pearson'
            )
            img_path = self._save_plot_as_image(fig, "correlation_matrix.png")
            
            if img_path.exists():
                img_elements = self.create_image(
                    img_path,
                    width=6 * inch,
                    caption="Figura: Matriz de Correlación (Pearson)"
                )
                content.extend(img_elements)
        except Exception as e:
            content.append(Paragraph(
                f"<i>Error al generar matriz de correlación: {str(e)}</i>",
                self.styles['CustomBody']
            ))
        
        content.append(Spacer(1, 0.2 * inch))
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= self.correlation_threshold:
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'corr': corr_val
                    })
        
        if strong_corr:
            content.append(Paragraph(
                f"<b>Correlaciones Fuertes Detectadas (|r| ≥ {self.correlation_threshold})</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            table_data = [['Variable 1', 'Variable 2', 'Correlación', 'Fuerza']]
            for item in sorted(strong_corr, key=lambda x: abs(x['corr']), reverse=True):
                strength = 'Fuerte positiva' if item['corr'] > 0 else 'Fuerte negativa'
                table_data.append([
                    item['var1'],
                    item['var2'],
                    f"{item['corr']:.4f}",
                    strength
                ])
            
            table = self.create_table(table_data, has_header=True)
            content.append(table)
        else:
            content.append(Paragraph(
                f"<i>No se encontraron correlaciones con |r| ≥ {self.correlation_threshold}</i>",
                self.styles['CustomBody']
            ))
        
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="2. Matriz de Correlación",
            level=1,
            content=content
        )
    
    def _create_pairwise_section(self, var1: str, var2: str, corr_value: float) -> ReportSection:
        """Create section for a significant pair of variables."""
        content = []
        
        # Bivariate statistics
        stats = self.analyze_numeric_numeric(self.df, var1, var2)
        
        content.append(Paragraph(
            "<b>Estadísticas de Correlación</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        stats_data = [
            ['Métrica', 'Valor', 'p-valor', 'Significancia'],
            [
                'Pearson',
                f"{stats.pearson_corr:.4f}" if stats.pearson_corr else "N/A",
                f"{stats.pearson_pvalue:.6f}" if stats.pearson_pvalue else "N/A",
                'Sí (α=0.05)' if stats.pearson_pvalue and stats.pearson_pvalue < 0.05 else 'No'
            ],
            [
                'Spearman',
                f"{stats.spearman_corr:.4f}" if stats.spearman_corr else "N/A",
                f"{stats.spearman_pvalue:.6f}" if stats.spearman_pvalue else "N/A",
                'Sí (α=0.05)' if stats.spearman_pvalue and stats.spearman_pvalue < 0.05 else 'No'
            ],
        ]
        
        table = self.create_table(stats_data, has_header=True)
        content.append(table)
        content.append(Spacer(1, 0.2 * inch))
        
        # Interpretation
        content.append(Paragraph(
            "<b>Interpretación</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        direction = "positiva" if corr_value > 0 else "negativa"
        content.append(Paragraph(
            f"Las variables <b>{var1}</b> y <b>{var2}</b> presentan una correlación "
            f"{stats.strength} {direction} (r = {corr_value:.4f}).",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        if corr_value > 0:
            content.append(Paragraph(
                "Esto indica que cuando una variable aumenta, la otra tiende a aumentar también.",
                self.styles['CustomBody']
            ))
        else:
            content.append(Paragraph(
                "Esto indica que cuando una variable aumenta, la otra tiende a disminuir.",
                self.styles['CustomBody']
            ))
        
        content.append(Spacer(1, 0.2 * inch))
        
        # Scatter plot
        content.append(Paragraph(
            "<b>Diagrama de Dispersión</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        try:
            fig = self.plot_scatter(self.df, var1, var2)
            img_path = self._save_plot_as_image(fig, f"scatter_{var1}_{var2}.png")
            
            if img_path.exists():
                img_elements = self.create_image(
                    img_path,
                    width=5.5 * inch,
                    caption=f"Figura: Relación entre {var1} y {var2}"
                )
                content.extend(img_elements)
        except Exception as e:
            content.append(Paragraph(
                f"<i>Error al generar scatter plot: {str(e)}</i>",
                self.styles['CustomBody']
            ))
        
        return ReportSection(
            title=f"{var1} vs {var2}",
            level=2,
            content=content,
            page_break_after=True
        )
    
    def _build_content(self) -> List[ReportSection]:
        """Build the report content."""
        sections = []
        
        # Overview
        sections.append(self._create_overview_section())
        
        # Correlation matrix
        sections.append(self._create_correlation_matrix_section())
        
        # Pairwise analysis for strong correlations
        df_numeric = self.df[self.numerical_cols].select_dtypes(include=[np.number])
        corr_matrix = df_numeric.corr()
        
        strong_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= self.correlation_threshold:
                    strong_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'corr': corr_val
                    })
        
        if strong_pairs:
            intro_content = [
                Paragraph(
                    f"Se identificaron {len(strong_pairs)} pares de variables con correlación "
                    f"fuerte (|r| ≥ {self.correlation_threshold}). A continuación se presenta "
                    "el análisis detallado de cada par.",
                    self.styles['CustomBody']
                ),
                Spacer(1, 0.2 * inch),
            ]
            
            sections.append(ReportSection(
                title="3. Análisis de Pares Significativos",
                level=1,
                content=intro_content,
                page_break_before=True
            ))
            
            # Add section for each strong pair
            for pair in sorted(strong_pairs, key=lambda x: abs(x['corr']), reverse=True):
                sections.append(self._create_pairwise_section(
                    pair['var1'],
                    pair['var2'],
                    pair['corr']
                ))
        
        # Conclusions
        conclusion_content = [
            Paragraph(
                "Este reporte ha analizado las relaciones bivariadas entre las variables numéricas "
                "del dataset, identificando correlaciones significativas.",
                self.styles['CustomBody']
            ),
            Spacer(1, 0.2 * inch),
            Paragraph(
                "<b>Hallazgos principales:</b>",
                self.styles['CustomHeading3']
            ),
            Spacer(1, 0.1 * inch),
            Paragraph(
                f"• Se analizaron {len(self.numerical_cols)} variables numéricas.",
                self.styles['CustomBody']
            ),
            Paragraph(
                f"• Se detectaron {len(strong_pairs)} pares con correlación |r| ≥ {self.correlation_threshold}.",
                self.styles['CustomBody']
            ),
        ]
        
        sections.append(ReportSection(
            title="4. Conclusiones",
            level=1,
            content=conclusion_content,
            page_break_before=True
        ))
        
        return sections


def generate_bivariate_pdf(
    df: pd.DataFrame,
    numerical_cols: List[str],
    output_path: str | Path,
    correlation_threshold: float = 0.7,
    template: Optional[ReportTemplate] = None,
    progress_callback: Optional[callable] = None
) -> Path:
    """
    Generate bivariate EDA PDF report.
    
    Args:
        df: DataFrame to analyze.
        numerical_cols: List of numerical column names.
        output_path: Path where to save the PDF.
        correlation_threshold: Minimum absolute correlation to include (default: 0.7).
        template: Optional report template configuration.
        progress_callback: Optional callback(progress: float, message: str).
    
    Returns:
        Path to generated PDF.
    
    Example:
        ```python
        from src.eda.pdf_reports import generate_bivariate_pdf
        
        pdf_path = generate_bivariate_pdf(
            df=data,
            numerical_cols=['age', 'bp', 'cholesterol'],
            output_path="reports/eda_bivariate.pdf",
            correlation_threshold=0.7
        )
        ```
    """
    report = BivariateEDAReport(
        df=df,
        numerical_cols=numerical_cols,
        correlation_threshold=correlation_threshold,
        template=template
    )
    
    return report.generate(
        output_path=output_path,
        async_mode=False,
        progress_callback=progress_callback
    )


# ============================================================================
# MULTIVARIATE ANALYSIS REPORT
# ============================================================================

class MultivariateEDAReport(PDFReportGenerator):
    """Generate PDF report for multivariate EDA."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        numerical_cols: List[str],
        n_components: Optional[int] = None,
        template: Optional[ReportTemplate] = None
    ):
        """
        Initialize multivariate EDA report.
        
        Args:
            df: DataFrame to analyze.
            numerical_cols: List of numerical column names.
            n_components: Number of PCA components (None = auto based on 95% variance).
            template: Report template configuration.
        """
        from src.eda.multivariate import perform_pca
        from src.eda.visualizations import plot_pca_scree, plot_pca_biplot, plot_correlation_matrix
        
        if template is None:
            template = ReportTemplate(
                title="Análisis Exploratorio de Datos",
                subtitle="Análisis Multivariado",
                author="Sistema Automatizado",
                organization="Mortality AMI Predictor",
                include_toc=True,
            )
        
        super().__init__(template)
        
        self.df = df
        self.numerical_cols = numerical_cols
        self.n_components = n_components
        self.perform_pca = perform_pca
        self.plot_pca_scree = plot_pca_scree
        self.plot_pca_biplot = plot_pca_biplot
        self.plot_correlation_matrix = plot_correlation_matrix
        
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
            f"<b>Fecha del análisis:</b> {datetime.now().strftime('%d de %B de %Y')}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        content.append(Paragraph(
            f"<b>Variables analizadas:</b> {len(self.numerical_cols)}",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        content.append(Paragraph(
            "Este reporte presenta un análisis multivariado completo, incluyendo:",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        content.append(Paragraph("• Matrices de correlación (Pearson, Spearman, Kendall)", self.styles['CustomBody']))
        content.append(Paragraph("• Análisis de Componentes Principales (PCA)", self.styles['CustomBody']))
        content.append(Paragraph("• Varianza explicada y loadings", self.styles['CustomBody']))
        content.append(Paragraph("• Biplot y Scree plot", self.styles['CustomBody']))
        content.append(Spacer(1, 0.2 * inch))
        
        return ReportSection(
            title="1. Resumen",
            level=1,
            content=content
        )
    
    def _create_correlation_section(self) -> ReportSection:
        """Create section with all correlation matrices."""
        content = []
        
        content.append(Paragraph(
            "Las matrices de correlación muestran diferentes medidas de asociación entre variables:",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.1 * inch))
        content.append(Paragraph(
            "• <b>Pearson:</b> Mide relaciones lineales entre variables.",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• <b>Spearman:</b> Mide relaciones monótonas (basado en rangos).",
            self.styles['CustomBody']
        ))
        content.append(Paragraph(
            "• <b>Kendall:</b> Otra medida basada en rangos, más robusta.",
            self.styles['CustomBody']
        ))
        content.append(Spacer(1, 0.2 * inch))
        
        # Pearson
        content.append(Paragraph(
            "<b>Correlación de Pearson</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        try:
            fig_pearson = self.plot_correlation_matrix(self.df, self.numerical_cols, method='pearson')
            img_path = self._save_plot_as_image(fig_pearson, "corr_pearson.png")
            
            if img_path.exists():
                img_elements = self.create_image(
                    img_path,
                    width=5.5 * inch,
                    caption="Figura: Matriz de Correlación de Pearson"
                )
                content.extend(img_elements)
        except Exception as e:
            content.append(Paragraph(f"<i>Error: {str(e)}</i>", self.styles['CustomBody']))
        
        # Spearman
        content.append(Paragraph(
            "<b>Correlación de Spearman</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        try:
            fig_spearman = self.plot_correlation_matrix(self.df, self.numerical_cols, method='spearman')
            img_path = self._save_plot_as_image(fig_spearman, "corr_spearman.png")
            
            if img_path.exists():
                img_elements = self.create_image(
                    img_path,
                    width=5.5 * inch,
                    caption="Figura: Matriz de Correlación de Spearman"
                )
                content.extend(img_elements)
        except Exception as e:
            content.append(Paragraph(f"<i>Error: {str(e)}</i>", self.styles['CustomBody']))
        
        # Kendall
        content.append(Paragraph(
            "<b>Correlación de Kendall</b>",
            self.styles['CustomHeading3']
        ))
        content.append(Spacer(1, 0.1 * inch))
        
        try:
            fig_kendall = self.plot_correlation_matrix(self.df, self.numerical_cols, method='kendall')
            img_path = self._save_plot_as_image(fig_kendall, "corr_kendall.png")
            
            if img_path.exists():
                img_elements = self.create_image(
                    img_path,
                    width=5.5 * inch,
                    caption="Figura: Matriz de Correlación de Kendall"
                )
                content.extend(img_elements)
        except Exception as e:
            content.append(Paragraph(f"<i>Error: {str(e)}</i>", self.styles['CustomBody']))
        
        return ReportSection(
            title="2. Matrices de Correlación",
            level=1,
            content=content,
            page_break_before=True
        )
    
    def _create_pca_section(self) -> ReportSection:
        """Create PCA analysis section."""
        content = []
        
        try:
            # Perform PCA
            pca_results = self.perform_pca(
                self.df,
                self.numerical_cols,
                n_components=self.n_components,
                variance_threshold=0.95,
                scale=True
            )
            
            content.append(Paragraph(
                "El Análisis de Componentes Principales (PCA) reduce la dimensionalidad del dataset "
                "identificando las direcciones de máxima varianza.",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.2 * inch))
            
            # Summary statistics
            content.append(Paragraph(
                "<b>Resumen del PCA</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            summary_data = [
                ['Métrica', 'Valor'],
                ['Número de componentes', f"{pca_results.n_components}"],
                ['Variables originales', f"{len(self.numerical_cols)}"],
                ['Varianza explicada (PC1)', f"{pca_results.explained_variance_ratio[0]*100:.2f}%"],
                ['Varianza acumulada', f"{pca_results.cumulative_variance[-1]*100:.2f}%"],
            ]
            
            table = self.create_table(summary_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
            
            # Variance table
            content.append(Paragraph(
                "<b>Varianza Explicada por Componente</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            var_data = [['Componente', 'Varianza', 'Varianza Acumulada']]
            for i in range(pca_results.n_components):
                var_data.append([
                    f"PC{i+1}",
                    f"{pca_results.explained_variance_ratio[i]*100:.2f}%",
                    f"{pca_results.cumulative_variance[i]*100:.2f}%"
                ])
            
            table = self.create_table(var_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
            
            # Scree plot
            content.append(Paragraph(
                "<b>Scree Plot</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            fig_scree = self.plot_pca_scree(pca_results)
            img_path = self._save_plot_as_image(fig_scree, "pca_scree.png")
            
            if img_path.exists():
                img_elements = self.create_image(
                    img_path,
                    width=5.5 * inch,
                    caption="Figura: Scree Plot - Varianza Explicada por Componente"
                )
                content.extend(img_elements)
            
            # Loadings table (top contributors to PC1 and PC2)
            content.append(Paragraph(
                "<b>Loadings - Principales Contribuyentes</b>",
                self.styles['CustomHeading3']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            content.append(Paragraph(
                "Los loadings indican cuánto contribuye cada variable original a cada componente principal.",
                self.styles['CustomBody']
            ))
            content.append(Spacer(1, 0.1 * inch))
            
            # PC1 top contributors
            pc1_loadings = [(pca_results.feature_names[i], pca_results.components[0, i]) 
                           for i in range(len(pca_results.feature_names))]
            pc1_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            
            load_data = [['Variable', 'PC1 Loading', 'PC2 Loading']]
            for i in range(min(10, len(pc1_loadings))):
                var_name = pc1_loadings[i][0]
                pc1_load = pc1_loadings[i][1]
                # Find PC2 loading for same variable
                var_idx = pca_results.feature_names.index(var_name)
                pc2_load = pca_results.components[1, var_idx] if pca_results.n_components > 1 else 0
                
                load_data.append([
                    var_name,
                    f"{pc1_load:.4f}",
                    f"{pc2_load:.4f}" if pca_results.n_components > 1 else "N/A"
                ])
            
            table = self.create_table(load_data, has_header=True)
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
            
            # Biplot
            if pca_results.n_components >= 2:
                content.append(Paragraph(
                    "<b>Biplot</b>",
                    self.styles['CustomHeading3']
                ))
                content.append(Spacer(1, 0.1 * inch))
                
                fig_biplot = self.plot_pca_biplot(pca_results, pc_x=1, pc_y=2)
                img_path = self._save_plot_as_image(fig_biplot, "pca_biplot.png")
                
                if img_path.exists():
                    img_elements = self.create_image(
                        img_path,
                        width=5.5 * inch,
                        caption="Figura: Biplot - PC1 vs PC2"
                    )
                    content.extend(img_elements)
            
        except Exception as e:
            content.append(Paragraph(
                f"<i>Error al realizar PCA: {str(e)}</i>",
                self.styles['CustomBody']
            ))
        
        return ReportSection(
            title="3. Análisis de Componentes Principales (PCA)",
            level=1,
            content=content,
            page_break_before=True
        )
    
    def _build_content(self) -> List[ReportSection]:
        """Build the report content."""
        sections = []
        
        # Overview
        sections.append(self._create_overview_section())
        
        # Correlation matrices
        sections.append(self._create_correlation_section())
        
        # PCA
        sections.append(self._create_pca_section())
        
        # Conclusions
        conclusion_content = [
            Paragraph(
                "Este reporte ha presentado un análisis multivariado completo del dataset, "
                "explorando relaciones complejas entre múltiples variables.",
                self.styles['CustomBody']
            ),
            Spacer(1, 0.2 * inch),
            Paragraph(
                "<b>Hallazgos principales:</b>",
                self.styles['CustomHeading3']
            ),
            Spacer(1, 0.1 * inch),
            Paragraph(
                f"• Se analizaron {len(self.numerical_cols)} variables numéricas.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Se generaron matrices de correlación con tres métodos diferentes.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• El PCA identificó los componentes principales que explican la mayor varianza.",
                self.styles['CustomBody']
            ),
            Spacer(1, 0.2 * inch),
            Paragraph(
                "<b>Recomendaciones:</b>",
                self.styles['CustomHeading3']
            ),
            Spacer(1, 0.1 * inch),
            Paragraph(
                "• Considerar usar los componentes principales para reducir dimensionalidad en modelado.",
                self.styles['CustomBody']
            ),
            Paragraph(
                "• Identificar y remover variables altamente correlacionadas para evitar multicolinealidad.",
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


def generate_multivariate_pdf(
    df: pd.DataFrame,
    numerical_cols: List[str],
    output_path: str | Path,
    n_components: Optional[int] = None,
    template: Optional[ReportTemplate] = None,
    progress_callback: Optional[callable] = None
) -> Path:
    """
    Generate multivariate EDA PDF report.
    
    Args:
        df: DataFrame to analyze.
        numerical_cols: List of numerical column names.
        output_path: Path where to save the PDF.
        n_components: Number of PCA components (None = auto based on 95% variance).
        template: Optional report template configuration.
        progress_callback: Optional callback(progress: float, message: str).
    
    Returns:
        Path to generated PDF.
    
    Example:
        ```python
        from src.eda.pdf_reports import generate_multivariate_pdf
        
        pdf_path = generate_multivariate_pdf(
            df=data,
            numerical_cols=['age', 'bp', 'cholesterol', 'glucose'],
            output_path="reports/eda_multivariate.pdf",
            n_components=None  # Auto-determine
        )
        ```
    """
    report = MultivariateEDAReport(
        df=df,
        numerical_cols=numerical_cols,
        n_components=n_components,
        template=template
    )
    
    return report.generate(
        output_path=output_path,
        async_mode=False,
        progress_callback=progress_callback
    )
