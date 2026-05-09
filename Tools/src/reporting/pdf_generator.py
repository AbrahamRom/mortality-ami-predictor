"""
PDF Report Generator Module.

This module provides a comprehensive framework for generating professional PDF reports
with support for async generation, progress tracking, and elegant formatting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import threading
import queue
from abc import ABC, abstractmethod
import io

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether
    )
    from reportlab.platypus.tableofcontents import TableOfContents
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


@dataclass
class ReportSection:
    """Represents a section in a PDF report."""
    
    title: str
    level: int = 1  # 1 for main section, 2 for subsection, etc.
    content: List[Any] = field(default_factory=list)  # List of Flowables
    page_break_before: bool = False
    page_break_after: bool = False
    
    def __post_init__(self):
        """Validate section."""
        if self.level < 1 or self.level > 3:
            raise ValueError("Section level must be between 1 and 3")


@dataclass
class ReportTemplate:
    """Configuration for PDF report appearance."""
    
    # Page settings
    page_size: Tuple[float, float] = A4
    top_margin: float = 0.75 * inch
    bottom_margin: float = 0.75 * inch
    left_margin: float = 0.75 * inch
    right_margin: float = 0.75 * inch
    
    # Colors
    primary_color: colors.Color = colors.HexColor("#1f77b4")
    secondary_color: colors.Color = colors.HexColor("#ff7f0e")
    header_bg_color: colors.Color = colors.HexColor("#f0f0f0")
    
    # Fonts
    title_font: str = "Helvetica-Bold"
    heading_font: str = "Helvetica-Bold"
    body_font: str = "Helvetica"
    
    # Font sizes
    title_size: int = 24
    heading1_size: int = 18
    heading2_size: int = 14
    heading3_size: int = 12
    body_size: int = 10
    
    # Report metadata
    title: str = "Report"
    subtitle: str = ""
    author: str = ""
    organization: str = "Mortality AMI Predictor"
    
    # Features
    include_toc: bool = True
    include_page_numbers: bool = True
    include_header: bool = True
    include_footer: bool = True
    
    def get_page_width(self) -> float:
        """Get usable page width."""
        return self.page_size[0] - self.left_margin - self.right_margin
    
    def get_page_height(self) -> float:
        """Get usable page height."""
        return self.page_size[1] - self.top_margin - self.bottom_margin


class PDFReportGenerator(ABC):
    """
    Abstract base class for generating PDF reports.
    
    This class provides a framework for creating professional PDF reports with:
    - Async generation support
    - Progress tracking
    - Elegant templates with headers/footers
    - Automatic table of contents
    - Image and table support
    - Multi-page handling
    
    Usage:
        class MyReport(PDFReportGenerator):
            def _build_content(self) -> List[ReportSection]:
                # Implement your report content
                pass
        
        report = MyReport(template)
        pdf_path = report.generate("output.pdf", progress_callback=callback)
    """
    
    def __init__(self, template: Optional[ReportTemplate] = None):
        """
        Initialize PDF report generator.
        
        Args:
            template: Report template configuration. If None, uses default.
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF generation. "
                "Install it with: pip install reportlab"
            )
        
        self.template = template or ReportTemplate()
        self.styles = self._create_styles()
        self.story = []  # List of flowables for the document
        self.toc = None
        
        # For async generation
        self._generation_thread = None
        self._progress_queue = queue.Queue()
        self._error_queue = queue.Queue()
        
    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles."""
        styles = getSampleStyleSheet()
        
        # Helper function to add style if it doesn't exist
        def add_style_if_not_exists(style_def):
            if style_def.name not in styles:
                styles.add(style_def)
        
        # Title style
        add_style_if_not_exists(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=self.template.title_size,
            textColor=self.template.primary_color,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName=self.template.title_font,
        ))
        
        # Subtitle style
        add_style_if_not_exists(ParagraphStyle(
            name='CustomSubtitle',
            parent=styles['Normal'],
            fontSize=self.template.heading2_size,
            textColor=self.template.secondary_color,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName=self.template.body_font,
        ))
        
        # Heading styles
        for level in [1, 2, 3]:
            size = getattr(self.template, f'heading{level}_size')
            add_style_if_not_exists(ParagraphStyle(
                name=f'CustomHeading{level}',
                parent=styles[f'Heading{level}'],
                fontSize=size,
                textColor=self.template.primary_color,
                spaceAfter=12,
                spaceBefore=12,
                fontName=self.template.heading_font,
            ))
        
        # Body text
        add_style_if_not_exists(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontSize=self.template.body_size,
            alignment=TA_JUSTIFY,
            fontName=self.template.body_font,
        ))
        
        # Code style
        add_style_if_not_exists(ParagraphStyle(
            name='Code',
            parent=styles['Normal'],
            fontSize=self.template.body_size - 1,
            fontName='Courier',
            backColor=colors.HexColor("#f5f5f5"),
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            spaceBefore=10,
        ))
        
        return styles
    
    def _add_header_footer(self, canvas, doc):
        """Add header and footer to each page."""
        canvas.saveState()
        
        # Header
        if self.template.include_header:
            canvas.setFont(self.template.body_font, 9)
            canvas.setFillColor(self.template.primary_color)
            canvas.drawString(
                self.template.left_margin,
                doc.pagesize[1] - 0.5 * inch,
                self.template.organization
            )
            canvas.drawRightString(
                doc.pagesize[0] - self.template.right_margin,
                doc.pagesize[1] - 0.5 * inch,
                datetime.now().strftime("%d/%m/%Y")
            )
            
            # Header line
            canvas.setStrokeColor(self.template.primary_color)
            canvas.setLineWidth(0.5)
            canvas.line(
                self.template.left_margin,
                doc.pagesize[1] - 0.6 * inch,
                doc.pagesize[0] - self.template.right_margin,
                doc.pagesize[1] - 0.6 * inch
            )
        
        # Footer
        if self.template.include_footer:
            canvas.setFont(self.template.body_font, 9)
            canvas.setFillColor(colors.grey)
            
            # Footer line
            canvas.setStrokeColor(colors.grey)
            canvas.setLineWidth(0.5)
            canvas.line(
                self.template.left_margin,
                0.6 * inch,
                doc.pagesize[0] - self.template.right_margin,
                0.6 * inch
            )
            
            # Page number
            if self.template.include_page_numbers:
                page_num = f"Página {doc.page}"
                canvas.drawRightString(
                    doc.pagesize[0] - self.template.right_margin,
                    0.5 * inch,
                    page_num
                )
            
            # Report title
            canvas.drawString(
                self.template.left_margin,
                0.5 * inch,
                self.template.title
            )
        
        canvas.restoreState()
    
    def _create_title_page(self) -> List[Any]:
        """Create the title page."""
        elements = []
        
        # Add spacing
        elements.append(Spacer(1, 2 * inch))
        
        # Title
        title = Paragraph(self.template.title, self.styles['CustomTitle'])
        elements.append(title)
        
        # Subtitle
        if self.template.subtitle:
            subtitle = Paragraph(self.template.subtitle, self.styles['CustomSubtitle'])
            elements.append(subtitle)
            elements.append(Spacer(1, 0.5 * inch))
        
        # Author and organization
        info_style = ParagraphStyle(
            name='Info',
            parent=self.styles['CustomBody'],
            alignment=TA_CENTER,
            fontSize=12,
        )
        
        if self.template.author:
            author = Paragraph(f"<b>Autor:</b> {self.template.author}", info_style)
            elements.append(author)
            elements.append(Spacer(1, 0.2 * inch))
        
        if self.template.organization:
            org = Paragraph(f"<b>Organización:</b> {self.template.organization}", info_style)
            elements.append(org)
            elements.append(Spacer(1, 0.3 * inch))
        
        # Date
        date = Paragraph(
            f"<b>Fecha:</b> {datetime.now().strftime('%d de %B de %Y')}",
            info_style
        )
        elements.append(date)
        
        # Page break
        elements.append(PageBreak())
        
        return elements
    
    def _create_toc(self) -> TableOfContents:
        """Create table of contents."""
        toc = TableOfContents()
        
        # Configure TOC style
        toc.levelStyles = [
            ParagraphStyle(
                name='TOCHeading1',
                fontSize=14,
                leftIndent=20,
                firstLineIndent=-20,
                spaceBefore=5,
                textColor=self.template.primary_color,
                fontName=self.template.heading_font,
            ),
            ParagraphStyle(
                name='TOCHeading2',
                fontSize=12,
                leftIndent=40,
                firstLineIndent=-20,
                spaceBefore=3,
                fontName=self.template.body_font,
            ),
            ParagraphStyle(
                name='TOCHeading3',
                fontSize=10,
                leftIndent=60,
                firstLineIndent=-20,
                spaceBefore=2,
                fontName=self.template.body_font,
            ),
        ]
        
        return toc
    
    def _add_section(self, section: ReportSection):
        """Add a section to the story."""
        # Page break before section if requested
        if section.page_break_before:
            self.story.append(PageBreak())
        
        # Add section heading
        style_name = f'CustomHeading{section.level}'
        heading = Paragraph(section.title, self.styles[style_name])
        self.story.append(heading)
        
        # Add to TOC if enabled
        if self.template.include_toc and self.toc:
            self.toc.addEntry(section.level - 1, section.title, len(self.story) - 1)
        
        # Add section content
        for item in section.content:
            self.story.append(item)
        
        # Page break after section if requested
        if section.page_break_after:
            self.story.append(PageBreak())
    
    @abstractmethod
    def _build_content(self) -> List[ReportSection]:
        """
        Build the report content.
        
        This method must be implemented by subclasses to define the actual
        content of the report.
        
        Returns:
            List of ReportSection objects to include in the report.
        """
        pass
    
    def _generate_pdf_internal(
        self,
        output_path: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """Internal method for generating PDF."""
        try:
            # Initialize story
            self.story = []
            
            # Update progress
            if progress_callback:
                progress_callback(0, "Inicializando reporte...")
            
            # Create title page
            if progress_callback:
                progress_callback(5, "Creando portada...")
            title_elements = self._create_title_page()
            self.story.extend(title_elements)
            
            # Create TOC
            if self.template.include_toc:
                if progress_callback:
                    progress_callback(10, "Creando índice...")
                self.toc = self._create_toc()
                self.story.append(self.toc)
                self.story.append(PageBreak())
            
            # Build content
            if progress_callback:
                progress_callback(20, "Construyendo contenido...")
            
            sections = self._build_content()
            
            # Add sections with progress updates
            for i, section in enumerate(sections):
                progress = 20 + (70 * (i + 1) / len(sections))
                if progress_callback:
                    progress_callback(progress, f"Agregando sección: {section.title}")
                self._add_section(section)
            
            # Build PDF
            if progress_callback:
                progress_callback(90, "Generando PDF...")
            
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=self.template.page_size,
                topMargin=self.template.top_margin,
                bottomMargin=self.template.bottom_margin,
                leftMargin=self.template.left_margin,
                rightMargin=self.template.right_margin,
            )
            
            # Build with header/footer
            doc.build(
                self.story,
                onFirstPage=self._add_header_footer,
                onLaterPages=self._add_header_footer
            )
            
            if progress_callback:
                progress_callback(100, "¡Reporte completado!")
            
            self._progress_queue.put((100, "completed", str(output_path)))
            
        except Exception as e:
            self._error_queue.put(e)
            if progress_callback:
                progress_callback(0, f"Error: {str(e)}")
    
    def generate(
        self,
        output_path: str | Path,
        async_mode: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Path:
        """
        Generate the PDF report.
        
        Args:
            output_path: Path where to save the PDF file.
            async_mode: If True, generate PDF in a separate thread.
            progress_callback: Optional callback function(progress: float, message: str)
                             called with progress updates (0-100).
        
        Returns:
            Path to the generated PDF file.
        
        Raises:
            Exception: If PDF generation fails.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if async_mode:
            # Generate in separate thread
            self._generation_thread = threading.Thread(
                target=self._generate_pdf_internal,
                args=(output_path, progress_callback)
            )
            self._generation_thread.start()
            return output_path
        else:
            # Generate synchronously
            self._generate_pdf_internal(output_path, progress_callback)
            return output_path
    
    def wait_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for async generation to complete.
        
        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.
        
        Returns:
            True if generation completed successfully, False if timeout.
        
        Raises:
            Exception: If generation failed.
        """
        if self._generation_thread is None:
            return True
        
        self._generation_thread.join(timeout)
        
        # Check for errors
        if not self._error_queue.empty():
            raise self._error_queue.get()
        
        return not self._generation_thread.is_alive()
    
    def get_progress(self) -> Optional[Tuple[float, str, Optional[str]]]:
        """
        Get current generation progress (for async mode).
        
        Returns:
            Tuple of (progress: float, status: str, output_path: Optional[str])
            or None if no progress available.
        """
        if self._progress_queue.empty():
            return None
        return self._progress_queue.get()
    
    # Utility methods for creating common elements
    
    def create_table(
        self,
        data: List[List[Any]],
        col_widths: Optional[List[float]] = None,
        has_header: bool = True,
        style: Optional[List[Tuple]] = None
    ) -> Table:
        """
        Create a formatted table.
        
        Args:
            data: 2D list of table data.
            col_widths: Optional list of column widths.
            has_header: If True, first row is styled as header.
            style: Optional custom TableStyle commands.
        
        Returns:
            Table object.
        """
        if style is None:
            style = [
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 0), (-1, -1), self.template.body_font),
                ('FONTSIZE', (0, 0), (-1, -1), self.template.body_size),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]
            
            if has_header:
                style.extend([
                    ('BACKGROUND', (0, 0), (-1, 0), self.template.header_bg_color),
                    ('FONTNAME', (0, 0), (-1, 0), self.template.heading_font),
                    ('TEXTCOLOR', (0, 0), (-1, 0), self.template.primary_color),
                ])
        
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle(style))
        
        return table
    
    def create_image(
        self,
        image_path: str | Path,
        width: Optional[float] = None,
        height: Optional[float] = None,
        caption: Optional[str] = None
    ) -> List[Any]:
        """
        Create an image with optional caption.
        
        Args:
            image_path: Path to image file.
            width: Image width in points. If None, uses available width.
            height: Image height in points. If None, maintains aspect ratio.
            caption: Optional caption text.
        
        Returns:
            List of flowables (image + caption).
        """
        elements = []
        
        # Use available width if not specified
        if width is None:
            width = self.template.get_page_width() * 0.8
        
        # Create image
        img = Image(str(image_path), width=width, height=height)
        elements.append(img)
        
        # Add caption
        if caption:
            caption_style = ParagraphStyle(
                name='Caption',
                parent=self.styles['CustomBody'],
                fontSize=self.template.body_size - 1,
                alignment=TA_CENTER,
                textColor=colors.grey,
                spaceBefore=5,
            )
            caption_p = Paragraph(f"<i>{caption}</i>", caption_style)
            elements.append(caption_p)
        
        elements.append(Spacer(1, 0.2 * inch))
        
        return elements
