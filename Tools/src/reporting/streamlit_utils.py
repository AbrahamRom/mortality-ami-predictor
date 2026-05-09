"""Streamlit utilities for PDF report generation.

This module provides helper functions to integrate PDF report generation
with Streamlit dashboards, including progress bars and download buttons.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Any
import streamlit as st
import threading

from src.reporting import PDFReportGenerator


def generate_pdf_with_progress(
    generator: PDFReportGenerator,
    output_path: str | Path,
    title: str = "Generando Reporte PDF",
    success_message: str = "¡Reporte generado exitosamente!",
    error_message: str = "Error al generar el reporte",
) -> Optional[Path]:
    """
    Generate a PDF report with a Streamlit progress bar.
    
    This function displays a progress bar while generating the PDF report
    and handles errors gracefully.
    
    Args:
        generator: PDFReportGenerator instance.
        output_path: Path where to save the PDF.
        title: Title to display above the progress bar.
        success_message: Message to display on success.
        error_message: Message to display on error.
    
    Returns:
        Path to generated PDF on success, None on error.
    
    Example:
        ```python
        from src.eda.pdf_reports import EDAReport
        
        report = EDAReport(data, plots)
        pdf_path = generate_pdf_with_progress(
            report,
            "reports/eda.pdf",
            title="Generando Reporte EDA"
        )
        
        if pdf_path:
            with open(pdf_path, "rb") as f:
                st.download_button("Descargar PDF", f, file_name="eda.pdf")
        ```
    """
    output_path = Path(output_path)
    
    # Create placeholders for progress
    progress_container = st.empty()
    status_container = st.empty()
    
    # Progress tracking variables
    progress_data = {"value": 0, "message": "Iniciando...", "completed": False, "error": None}
    
    def progress_callback(progress: float, message: str):
        """Update progress."""
        progress_data["value"] = progress
        progress_data["message"] = message
    
    try:
        # Display title
        status_container.info(f"🔄 {title}...")
        
        # Generate PDF in background thread
        generation_thread = threading.Thread(
            target=lambda: generator.generate(
                output_path,
                async_mode=False,
                progress_callback=progress_callback
            )
        )
        generation_thread.start()
        
        # Update progress bar
        while generation_thread.is_alive():
            progress = progress_data["value"]
            message = progress_data["message"]
            
            progress_container.progress(
                int(progress) / 100,
                text=f"{message} ({progress:.1f}%)"
            )
            
            time.sleep(0.1)
        
        generation_thread.join()
        
        # Final update
        progress_container.progress(100, text="Completado (100%)")
        
        # Check if file was created
        if output_path.exists():
            status_container.success(f"✅ {success_message}")
            return output_path
        else:
            status_container.error(f"❌ {error_message}: Archivo no creado")
            return None
            
    except Exception as e:
        progress_container.empty()
        status_container.error(f"❌ {error_message}: {str(e)}")
        return None


def create_pdf_download_button(
    pdf_path: str | Path,
    button_label: str = "📄 Descargar PDF",
    file_name: str = "reporte.pdf",
    mime: str = "application/pdf",
    key: Optional[str] = None,
    help: Optional[str] = None,
) -> bool:
    """
    Create a Streamlit download button for a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        button_label: Label for the download button.
        file_name: Name for the downloaded file.
        mime: MIME type (default: application/pdf).
        key: Optional unique key for the button.
        help: Optional help text to display on hover.
    
    Returns:
        True if button was clicked, False otherwise.
    
    Example:
        ```python
        if pdf_path:
            create_pdf_download_button(
                pdf_path,
                button_label="📄 Descargar Reporte EDA",
                file_name="reporte_eda.pdf",
                help="Descarga el reporte completo de análisis exploratorio"
            )
        ```
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        st.error(f"❌ Archivo no encontrado: {pdf_path}")
        return False
    
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        
        return st.download_button(
            label=button_label,
            data=pdf_bytes,
            file_name=file_name,
            mime=mime,
            key=key,
            help=help,
        )
    except Exception as e:
        st.error(f"❌ Error al leer el archivo PDF: {str(e)}")
        return False


def pdf_export_section(
    generate_callback: Callable[[], Optional[Path]],
    section_title: str = "Exportar Reporte PDF",
    button_label: str = "🚀 Generar PDF",
    download_label: str = "📄 Descargar PDF",
    default_filename: str = "reporte.pdf",
    description: Optional[str] = None,
    key_prefix: str = "pdf_export",
) -> None:
    """
    Create a complete PDF export section with generation and download.
    
    This function creates a UI section with:
    - Optional description text
    - Generate button
    - Progress tracking during generation
    - Download button when ready
    
    Args:
        generate_callback: Function that generates the PDF and returns the path.
        section_title: Title of the section.
        button_label: Label for the generate button.
        download_label: Label for the download button.
        default_filename: Default filename for download.
        description: Optional description text.
        key_prefix: Prefix for widget keys (must be unique per section).
    
    Example:
        ```python
        def generate_eda_pdf():
            report = EDAReport(data, plots)
            return generate_pdf_with_progress(
                report, 
                "reports/eda.pdf",
                title="Generando Reporte EDA"
            )
        
        pdf_export_section(
            generate_eda_pdf,
            section_title="Exportar Análisis Exploratorio",
            description="Genera un reporte PDF completo del EDA univariado",
            default_filename="eda_univariado.pdf",
            key_prefix="eda_univariate"
        )
        ```
    """
    st.subheader(section_title)
    
    if description:
        st.markdown(description)
    
    # Check if PDF already exists in session state
    pdf_path_key = f"{key_prefix}_pdf_path"
    
    # Generate button
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button(button_label, key=f"{key_prefix}_generate"):
            # Clear previous PDF path
            if pdf_path_key in st.session_state:
                del st.session_state[pdf_path_key]
            
            # Generate new PDF
            pdf_path = generate_callback()
            
            if pdf_path:
                st.session_state[pdf_path_key] = str(pdf_path)
    
    # Download button (if PDF exists)
    if pdf_path_key in st.session_state:
        pdf_path = Path(st.session_state[pdf_path_key])
        
        if pdf_path.exists():
            with col2:
                create_pdf_download_button(
                    pdf_path,
                    button_label=download_label,
                    file_name=default_filename,
                    key=f"{key_prefix}_download",
                    help=f"Descargar {default_filename}"
                )
        else:
            st.warning("⚠️ El archivo PDF ya no existe. Genera uno nuevo.")
            del st.session_state[pdf_path_key]
    
    st.divider()


def batch_pdf_export(
    generators: dict[str, Callable[[], Optional[Path]]],
    section_title: str = "Exportación por Lotes",
    description: Optional[str] = None,
) -> None:
    """
    Create a section for generating multiple PDFs at once.
    
    Args:
        generators: Dict mapping report names to generator callbacks.
        section_title: Title of the section.
        description: Optional description text.
    
    Example:
        ```python
        generators = {
            "EDA Univariado": lambda: generate_eda_univariate(),
            "EDA Bivariado": lambda: generate_eda_bivariate(),
            "EDA Multivariado": lambda: generate_eda_multivariate(),
        }
        
        batch_pdf_export(
            generators,
            section_title="Exportar Todos los Reportes EDA",
            description="Genera todos los reportes de análisis exploratorio"
        )
        ```
    """
    st.subheader(section_title)
    
    if description:
        st.markdown(description)
    
    # Selection
    selected_reports = st.multiselect(
        "Selecciona los reportes a generar:",
        options=list(generators.keys()),
        default=list(generators.keys()),
        key="batch_pdf_selection"
    )
    
    if st.button("🚀 Generar Reportes Seleccionados", key="batch_generate"):
        if not selected_reports:
            st.warning("⚠️ No has seleccionado ningún reporte")
            return
        
        st.info(f"📊 Generando {len(selected_reports)} reportes...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        for i, report_name in enumerate(selected_reports):
            status_text.text(f"Generando: {report_name}...")
            
            try:
                pdf_path = generators[report_name]()
                results[report_name] = pdf_path
            except Exception as e:
                st.error(f"❌ Error generando {report_name}: {str(e)}")
                results[report_name] = None
            
            progress_bar.progress((i + 1) / len(selected_reports))
        
        status_text.empty()
        progress_bar.empty()
        
        # Show results
        successful = [name for name, path in results.items() if path]
        failed = [name for name, path in results.items() if not path]
        
        if successful:
            st.success(f"✅ {len(successful)} reportes generados exitosamente")
            
            # Create download buttons
            for report_name in successful:
                pdf_path = results[report_name]
                filename = f"{report_name.lower().replace(' ', '_')}.pdf"
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.text(f"📄 {report_name}")
                with col2:
                    create_pdf_download_button(
                        pdf_path,
                        button_label="Descargar",
                        file_name=filename,
                        key=f"batch_download_{report_name}"
                    )
        
        if failed:
            st.warning(f"⚠️ {len(failed)} reportes fallaron: {', '.join(failed)}")
    
    st.divider()
