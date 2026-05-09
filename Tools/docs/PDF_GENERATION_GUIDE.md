# Sistema de Generación de Reportes PDF

Sistema completo para generar reportes PDF profesionales con soporte para generación asíncrona, barras de progreso, y templates elegantes.

## Características

- ✅ **Generación asíncrona** con threading para no bloquear el UI
- ✅ **Barra de progreso** con callbacks personalizables
- ✅ **Templates elegantes** con encabezados, pie de página y colores personalizables
- ✅ **Índice automático** (Table of Contents)
- ✅ **Soporte para imágenes y tablas** con estilos profesionales
- ✅ **Manejo automático de múltiples páginas**
- ✅ **Integración con Streamlit** mediante utilidades especializadas

## Arquitectura

```
src/reporting/
├── __init__.py                 # Exports del módulo
├── pdf_generator.py            # Clase base PDFReportGenerator
└── streamlit_utils.py          # Utilidades para integración con Streamlit
```

## Uso Básico

### 1. Crear una Clase de Reporte Personalizado

```python
from typing import List
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.units import inch

from src.reporting import PDFReportGenerator, ReportSection, ReportTemplate


class MyCustomReport(PDFReportGenerator):
    """Reporte personalizado."""
    
    def __init__(self, data: dict, template: ReportTemplate = None):
        super().__init__(template)
        self.data = data
    
    def _build_content(self) -> List[ReportSection]:
        """Construir el contenido del reporte."""
        sections = []
        
        # Sección 1: Introducción
        intro_content = [
            Paragraph("Este es el texto de introducción.", self.styles['CustomBody']),
            Spacer(1, 0.2 * inch),
        ]
        
        sections.append(ReportSection(
            title="Introducción",
            level=1,
            content=intro_content
        ))
        
        # Sección 2: Tabla de Resultados
        table_data = [
            ['Métrica', 'Valor'],
            ['AUROC', '0.85'],
            ['Precisión', '0.82'],
        ]
        
        table_content = [
            self.create_table(table_data, has_header=True),
            Spacer(1, 0.2 * inch),
        ]
        
        sections.append(ReportSection(
            title="Resultados",
            level=1,
            content=table_content
        ))
        
        return sections
```

### 2. Generar el PDF

#### Generación Síncrona (con callback de progreso)

```python
from pathlib import Path

# Configurar template
template = ReportTemplate(
    title="Mi Reporte",
    subtitle="Análisis de Datos",
    author="Tu Nombre",
    organization="Mortality AMI Predictor",
    include_toc=True,
)

# Callback para tracking
def progress_callback(progress: float, message: str):
    print(f"[{progress:5.1f}%] {message}")

# Generar reporte
report = MyCustomReport(data=my_data, template=template)
pdf_path = report.generate(
    output_path="reports/mi_reporte.pdf",
    async_mode=False,
    progress_callback=progress_callback
)

print(f"✅ PDF generado: {pdf_path}")
```

#### Generación Asíncrona (en background)

```python
# Generar en background
report = MyCustomReport(data=my_data, template=template)
report.generate(
    output_path="reports/mi_reporte.pdf",
    async_mode=True,
    progress_callback=progress_callback
)

print("Generación en progreso...")
# Hacer otras cosas mientras se genera...

# Esperar a que termine
success = report.wait_completion(timeout=30)
if success:
    print("✅ Generación completada!")
```

## Integración con Streamlit

### Opción 1: Generación con Barra de Progreso

```python
import streamlit as st
from src.reporting import generate_pdf_with_progress

# En tu página de Streamlit
if st.button("Generar Reporte"):
    report = MyCustomReport(data=my_data, template=template)
    
    pdf_path = generate_pdf_with_progress(
        report,
        output_path="reports/mi_reporte.pdf",
        title="Generando Reporte de Análisis",
        success_message="¡Reporte generado exitosamente!",
    )
    
    if pdf_path:
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Descargar PDF",
                f,
                file_name="mi_reporte.pdf",
                mime="application/pdf"
            )
```

### Opción 2: Sección Completa de Exportación

```python
from src.reporting import pdf_export_section

def generate_my_report():
    """Función que genera el reporte y retorna el path."""
    report = MyCustomReport(data=my_data, template=template)
    return generate_pdf_with_progress(
        report,
        "reports/mi_reporte.pdf",
        title="Generando Reporte"
    )

# Crear sección de exportación completa
pdf_export_section(
    generate_callback=generate_my_report,
    section_title="Exportar Reporte de Análisis",
    button_label="🚀 Generar PDF",
    download_label="📄 Descargar Reporte",
    default_filename="analisis_completo.pdf",
    description="Genera un reporte PDF con todos los resultados del análisis.",
    key_prefix="my_report"  # Debe ser único
)
```

### Opción 3: Exportación por Lotes

```python
from src.reporting import batch_pdf_export

# Definir generadores para cada tipo de reporte
generators = {
    "Reporte EDA": lambda: generate_eda_report(),
    "Reporte de Training": lambda: generate_training_report(),
    "Reporte de Evaluación": lambda: generate_evaluation_report(),
}

# Crear sección de exportación por lotes
batch_pdf_export(
    generators,
    section_title="Exportar Todos los Reportes",
    description="Genera múltiples reportes simultáneamente"
)
```

## Personalización de Templates

```python
from reportlab.lib import colors

# Template personalizado
custom_template = ReportTemplate(
    # Configuración de página
    page_size=A4,  # o letter
    top_margin=0.75 * inch,
    bottom_margin=0.75 * inch,
    left_margin=1 * inch,
    right_margin=1 * inch,
    
    # Colores personalizados
    primary_color=colors.HexColor("#2E7D32"),  # Verde oscuro
    secondary_color=colors.HexColor("#FF6F00"),  # Naranja
    header_bg_color=colors.HexColor("#E8F5E9"),  # Verde claro
    
    # Fuentes
    title_font="Helvetica-Bold",
    heading_font="Helvetica-Bold",
    body_font="Helvetica",
    
    # Tamaños
    title_size=24,
    heading1_size=18,
    heading2_size=14,
    body_size=10,
    
    # Metadata
    title="Mi Reporte Personalizado",
    subtitle="Con Estilos Custom",
    author="Equipo de ML",
    organization="Mi Organización",
    
    # Características
    include_toc=True,
    include_page_numbers=True,
    include_header=True,
    include_footer=True,
)
```

## Métodos Útiles de la Clase Base

### Crear Tablas

```python
def _build_content(self) -> List[ReportSection]:
    # Datos de la tabla
    table_data = [
        ['Columna 1', 'Columna 2', 'Columna 3'],  # Header
        ['Dato 1', 'Dato 2', 'Dato 3'],
        ['Dato 4', 'Dato 5', 'Dato 6'],
    ]
    
    # Crear tabla con estilo
    table = self.create_table(
        data=table_data,
        col_widths=[2*inch, 2*inch, 2*inch],  # opcional
        has_header=True,
        style=None  # o lista de comandos TableStyle personalizados
    )
    
    content = [table]
    # ...
```

### Agregar Imágenes

```python
def _build_content(self) -> List[ReportSection]:
    # Crear imagen con caption
    image_elements = self.create_image(
        image_path="path/to/image.png",
        width=5*inch,  # opcional, por defecto usa 80% del ancho disponible
        height=None,   # opcional, mantiene aspect ratio
        caption="Figura 1: Descripción de la imagen"
    )
    
    content = image_elements  # Ya es una lista
    # ...
```

### Estilos Disponibles

```python
# En tu método _build_content, puedes usar:
self.styles['CustomTitle']      # Título principal
self.styles['CustomSubtitle']   # Subtítulo
self.styles['CustomHeading1']   # Encabezado nivel 1
self.styles['CustomHeading2']   # Encabezado nivel 2
self.styles['CustomHeading3']   # Encabezado nivel 3
self.styles['CustomBody']       # Texto normal
self.styles['Code']             # Texto tipo código
```

## Estructura de ReportSection

```python
from src.reporting import ReportSection

section = ReportSection(
    title="Título de la Sección",
    level=1,  # 1, 2, o 3 (principal, sub, subsub)
    content=[  # Lista de Flowables de ReportLab
        Paragraph("Texto...", styles['CustomBody']),
        Spacer(1, 0.2 * inch),
        table,
        image,
    ],
    page_break_before=False,  # Salto de página antes
    page_break_after=False,   # Salto de página después
)
```

## Próximos Módulos de Reportes

El sistema está diseñado para ser extendido. Los siguientes módulos implementarán reportes específicos:

1. **`src/eda/pdf_reports.py`**: Reportes de EDA (univariado, bivariado, multivariado)
2. **`src/training/pdf_reports.py`**: Reportes de entrenamiento (hiperparámetros, CV, comparaciones)
3. **`src/evaluation/pdf_reports.py`**: Reportes de evaluación (métricas, curvas ROC/PR, matrices)
4. **`src/explainability/pdf_reports.py`**: Reportes de explicabilidad (SHAP, feature importance)

Cada módulo extenderá `PDFReportGenerator` y usará las utilidades de Streamlit para integración con el dashboard.

## Ejemplo Completo

Ver `examples/pdf_report_usage.py` para ejemplos completos de uso.

## Troubleshooting

### Error: "ReportLab is required"
```bash
pip install reportlab
```

### Imágenes no se muestran
- Verifica que el path de la imagen sea correcto
- Asegúrate de que el formato sea soportado (PNG, JPG, etc.)
- Usa paths absolutos o relativos correctos

### Tablas se cortan entre páginas
- Usa `KeepTogether()` de ReportLab:
```python
from reportlab.platypus import KeepTogether

content = [
    KeepTogether([table, caption])
]
```

### Texto se sale de los márgenes
- Ajusta los anchos de columna en tablas
- Reduce el tamaño de fuente
- Aumenta los márgenes del template

## Recursos Adicionales

- [Documentación de ReportLab](https://www.reportlab.com/docs/reportlab-userguide.pdf)
- [Streamlit Download Button](https://docs.streamlit.io/library/api-reference/widgets/st.download_button)
- [Python Threading](https://docs.python.org/3/library/threading.html)
