import os
import base64
import io
from datetime import datetime
from typing import Dict, List
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

class MedicalReportGenerator:
    def __init__(self):
        self.page_width, self.page_height = letter
        self.setup_styles()
    
    def setup_styles(self):
        """Setup custom styles for the medical report"""
        self.styles = getSampleStyleSheet()
        
        # Company header style
        self.styles.add(ParagraphStyle(
            name='CompanyHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=HexColor('#1e40af'),
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName='Helvetica-Bold'
        ))
        
        # Service title style
        self.styles.add(ParagraphStyle(
            name='ServiceTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#1e40af'),
            alignment=TA_CENTER,
            spaceAfter=15,
            fontName='Helvetica-Bold'
        ))
        
        # Report title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=black,
            alignment=TA_CENTER,
            spaceAfter=25,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=HexColor('#1e40af'),
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
        
        # Normal text style
        self.styles.add(ParagraphStyle(
            name='NormalText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=black,
            spaceAfter=6,
            fontName='Helvetica'
        ))
        
        # Patient info style
        self.styles.add(ParagraphStyle(
            name='PatientInfo',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=black,
            spaceAfter=4,
            fontName='Helvetica'
        ))
        
        # Results style
        self.styles.add(ParagraphStyle(
            name='ResultsText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=black,
            spaceAfter=3,
            fontName='Helvetica'
        ))

    def create_report(self, 
                     analysis_data: Dict,
                     image_data: str,
                     patient_name: str = "Mr Ramzi Houidi") -> bytes:
        """
        Create a medical report PDF
        
        Args:
            analysis_data: Dictionary containing analysis results
            image_data: Base64 encoded image data
            patient_name: Name of the patient
            
        Returns:
            PDF file as bytes
        """
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Build the story (content)
        story = []
        
        # Add company header
        story.extend(self._create_header())
        
        # Add report title
        story.extend(self._create_report_title())
        
        # Add patient information
        story.extend(self._create_patient_info(patient_name))
        
        # Add analysis image
        story.extend(self._create_image_section(image_data))
        
        # Add analysis results
        story.extend(self._create_results_section(analysis_data))
        
        # Add recommendations
        story.extend(self._create_recommendations_section(analysis_data))
        
        # Add product recommendations
        story.extend(self._create_products_section(analysis_data))
        
        # Add disclaimer
        story.extend(self._create_disclaimer())
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes

    def _create_header(self) -> List:
        """Create company header section"""
        elements = []
        
        # Company name
        elements.append(Paragraph("Inveep Inc", self.styles['CompanyHeader']))
        
        # Service name
        elements.append(Paragraph("MedicImage - DermaScan", self.styles['ServiceTitle']))
        
        # Date and time
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        elements.append(Paragraph(f"Report Generated: {current_datetime}", self.styles['NormalText']))
        
        elements.append(Spacer(1, 20))
        return elements

    def _create_report_title(self) -> List:
        """Create report title section"""
        elements = []
        
        elements.append(Paragraph("SKIN ANALYSIS REPORT", self.styles['ReportTitle']))
        elements.append(Spacer(1, 15))
        
        return elements

    def _create_patient_info(self, patient_name: str) -> List:
        """Create patient information section"""
        elements = []
        
        elements.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeader']))
        
        # Patient info table
        patient_data = [
            ["Patient Name:", patient_name],
            ["Report Date:", datetime.now().strftime("%B %d, %Y")],
            ["Report Time:", datetime.now().strftime("%I:%M %p")],
            ["Analysis Type:", "AI-Powered Skin Condition Classification"]
        ]
        
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#f3f4f6')),
        ]))
        
        elements.append(patient_table)
        elements.append(Spacer(1, 15))
        
        return elements

    def _create_image_section(self, image_data: str) -> List:
        """Create image section"""
        elements = []
        
        elements.append(Paragraph("ANALYZED IMAGE", self.styles['SectionHeader']))
        
        try:
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            pil_image = PILImage.open(io.BytesIO(image_bytes))
            
            # Resize image to fit on page (max width 4 inches)
            max_width = 4 * inch
            max_height = 3 * inch
            
            # Calculate aspect ratio
            aspect_ratio = pil_image.width / pil_image.height
            
            if aspect_ratio > 1:  # Landscape
                width = min(max_width, pil_image.width)
                height = width / aspect_ratio
            else:  # Portrait
                height = min(max_height, pil_image.height)
                width = height * aspect_ratio
            
            # Convert PIL image to ReportLab image
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            reportlab_image = Image(img_buffer, width=width, height=height)
            reportlab_image.hAlign = 'CENTER'
            
            elements.append(reportlab_image)
            
        except Exception as e:
            elements.append(Paragraph(f"Image could not be processed: {str(e)}", self.styles['NormalText']))
        
        elements.append(Spacer(1, 15))
        return elements

    def _create_results_section(self, analysis_data: Dict) -> List:
        """Create analysis results section"""
        elements = []
        
        elements.append(Paragraph("ANALYSIS RESULTS", self.styles['SectionHeader']))
        
        # Primary condition
        primary_condition = analysis_data.get('primary_condition', 'Unknown')
        confidence = analysis_data.get('confidence', 0)
        
        elements.append(Paragraph(
            f"<b>Primary Condition:</b> {primary_condition}",
            self.styles['NormalText']
        ))
        elements.append(Paragraph(
            f"<b>Confidence Level:</b> {confidence:.1%}",
            self.styles['NormalText']
        ))
        
        elements.append(Spacer(1, 10))
        
        # All condition probabilities
        predictions = analysis_data.get('predictions', {})
        if predictions:
            elements.append(Paragraph("<b>Condition Probabilities:</b>", self.styles['NormalText']))
            
            # Create results table
            results_data = [["Condition", "Probability", "Severity"]]
            
            for condition, probability in predictions.items():
                severity = "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low"
                results_data.append([
                    condition,
                    f"{probability:.1%}",
                    severity
                ])
            
            results_table = Table(results_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
            results_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ]))
            
            elements.append(results_table)
        
        elements.append(Spacer(1, 15))
        return elements

    def _create_recommendations_section(self, analysis_data: Dict) -> List:
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("RECOMMENDATIONS", self.styles['SectionHeader']))
        
        recommendations = analysis_data.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                elements.append(Paragraph(
                    f"{i}. {rec}",
                    self.styles['NormalText']
                ))
        else:
            elements.append(Paragraph(
                "Please consult with a dermatologist for personalized recommendations.",
                self.styles['NormalText']
            ))
        
        elements.append(Spacer(1, 15))
        return elements

    def _create_products_section(self, analysis_data: Dict) -> List:
        """Create product recommendations section"""
        elements = []
        
        elements.append(Paragraph("RECOMMENDED PRODUCTS", self.styles['SectionHeader']))
        
        products = analysis_data.get('products', [])
        if products:
            # Create products table
            products_data = [["Product", "Brand", "Rating", "Price"]]
            
            for product in products[:5]:  # Limit to 5 products
                products_data.append([
                    product.get('name', 'N/A'),
                    product.get('brand', 'N/A'),
                    f"{product.get('rating', 0)} ★",
                    product.get('price', 'N/A')
                ])
            
            products_table = Table(products_data, colWidths=[2*inch, 1.5*inch, 0.8*inch, 1*inch])
            products_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#059669')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ]))
            
            elements.append(products_table)
        else:
            elements.append(Paragraph(
                "Product recommendations will be available based on your specific condition.",
                self.styles['NormalText']
            ))
        
        elements.append(Spacer(1, 15))
        return elements

    def _create_disclaimer(self) -> List:
        """Create medical disclaimer section"""
        elements = []
        
        elements.append(Paragraph("IMPORTANT DISCLAIMER", self.styles['SectionHeader']))
        
        disclaimer_text = """
        This report is generated by an AI-powered skin analysis tool for educational and informational purposes only. 
        The results shown are approximations based on AI analysis and should not be considered as definitive medical diagnoses.
        
        This tool is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult with a qualified dermatologist or healthcare provider for accurate diagnosis and appropriate treatment.
        
        The confidence percentages indicate the model's certainty in classification, not medical accuracy. 
        Results may vary and should not be used for self-diagnosis or treatment decisions.
        
        If you have concerns about your skin condition, please schedule an appointment with a dermatologist 
        for proper evaluation and treatment.
        """
        
        elements.append(Paragraph(disclaimer_text, self.styles['NormalText']))
        
        # Footer
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(
            "Generated by MedicImage DermaScan - Inveep Inc",
            ParagraphStyle(
                'Footer',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=HexColor('#6b7280'),
                alignment=TA_CENTER,
                fontName='Helvetica'
            )
        ))
        
        return elements 