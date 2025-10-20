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

class BrainTumorReportGenerator:
    def __init__(self):
        self.page_width, self.page_height = letter
        self.setup_styles()
    
    def setup_styles(self):
        """Setup custom styles for the brain tumor medical report"""
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

    def create_report(self, 
                     analysis_data: Dict,
                     patient_name: str = "Anonyme") -> bytes:
        """
        Create a brain tumor medical report PDF
        
        Args:
            analysis_data: Dictionary containing analysis results
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
        
        # Add analysis results
        story.extend(self._create_results_section(analysis_data))
        
        # Add segmentation image if available
        if 'segmentation_image_url' in analysis_data:
            story.extend(self._create_segmentation_image_section(analysis_data))
        
        # Add survival prediction if available
        if 'survival_prediction' in analysis_data:
            story.extend(self._create_survival_section(analysis_data))
        
        # Add extracted features if available
        if 'extracted_features' in analysis_data:
            story.extend(self._create_features_section(analysis_data))
        
        # Add recommendations
        story.extend(self._create_recommendations_section(analysis_data))
        
        # Add disclaimer
        story.extend(self._create_disclaimer(analysis_data))
        
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
        elements.append(Paragraph("MedicImage - Brain Tumor Analysis", self.styles['ServiceTitle']))
        
        # Date and time
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        elements.append(Paragraph(f"Report Generated: {current_datetime}", self.styles['NormalText']))
        
        elements.append(Spacer(1, 20))
        return elements

    def _create_report_title(self) -> List:
        """Create report title section"""
        elements = []
        
        elements.append(Paragraph("BRAIN TUMOR ANALYSIS REPORT", self.styles['ReportTitle']))
        elements.append(Spacer(1, 15))
        
        return elements

    def _create_patient_info(self, patient_name: str) -> List:
        """Create patient information section"""
        elements = []
        
        elements.append(Paragraph("PATIENT INFORMATION", self.styles['SectionHeader']))
        
        # Patient info table
        patient_data = [
            ["Patient name:", patient_name],
            ["Report date:", datetime.now().strftime("%B %d, %Y")],
            ["Report time:", datetime.now().strftime("%H:%M")],
            ["Analysis type:", "Brain Tumor Segmentation & AI Survival Prediction"]
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

    def _create_results_section(self, analysis_data: Dict) -> List:
        """Create primary analysis results section"""
        elements = []
        
        elements.append(Paragraph("PRIMARY ANALYSIS RESULTS", self.styles['SectionHeader']))
        
        # Primary tumor type
        tumor_type = analysis_data.get('tumor_type', 'Unknown')
        
        # 🎯 CORRECTION: Retrieve confidence from different possible sources
        confidence = 0
        
        # Method 1: direct confidence
        if 'confidence' in analysis_data and analysis_data['confidence'] is not None:
            confidence = analysis_data['confidence']
        
        # Method 2: from survival_prediction
        elif 'survival_prediction' in analysis_data and isinstance(analysis_data['survival_prediction'], dict):
            if 'confidence' in analysis_data['survival_prediction']:
                confidence = analysis_data['survival_prediction']['confidence']
        
        # Method 3: from segmentation_info  
        elif 'segmentation_info' in analysis_data and isinstance(analysis_data['segmentation_info'], dict):
            if 'confidence' in analysis_data['segmentation_info']:
                confidence = analysis_data['segmentation_info']['confidence']
        
        # Method 4: fallback - search throughout the structure
        else:
            # Search recursively in all sub-dictionaries
            def find_confidence(data, depth=0):
                if depth > 3:  # Limit depth to avoid loops
                    return None
                    
                if isinstance(data, dict):
                    # Search for confidence directly
                    for key in ['confidence', 'detection_confidence', 'model_confidence']:
                        if key in data and isinstance(data[key], (int, float)) and data[key] > 0:
                            return data[key]
                    
                    # Search recursively
                    for value in data.values():
                        result = find_confidence(value, depth + 1)
                        if result is not None:
                            return result
                
                return None
            
            found_confidence = find_confidence(analysis_data)
            if found_confidence is not None:
                confidence = found_confidence
        
        # Ensure confidence is in a reasonable range
        if confidence > 1.0:
            confidence = confidence / 100.0  # Convert percentage (85.4) to decimal (0.854)
        
        elements.append(Paragraph(
            f"<b>Detected tumor type:</b> {tumor_type}",
            self.styles['NormalText']
        ))
        elements.append(Paragraph(
            f"<b>Detection confidence:</b> {confidence:.1%}",
            self.styles['NormalText']
        ))
        
        # Analysis success status
        success = analysis_data.get('success', False)
        elements.append(Paragraph(
            f"<b>Analysis status:</b> {'Successfully completed' if success else 'Partial analysis'}",
            self.styles['NormalText']
        ))
        
        elements.append(Spacer(1, 10))
        return elements

    def _create_segmentation_image_section(self, analysis_data: Dict) -> List:
        """Create segmentation image section"""
        elements = []
        
        elements.append(Paragraph("SEGMENTATION IMAGE", self.styles['SectionHeader']))
        
        segmentation_image_url = analysis_data.get('segmentation_image_url', '')
        
        if segmentation_image_url:
            try:
                # Build the complete image path
                if segmentation_image_url.startswith('/'):
                    # Relative path from server - remove slash and search in backend directory
                    image_filename = segmentation_image_url.lstrip('/')
                    if image_filename == 'get_segmentation_image':
                        # Use default filename
                        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segmentation_result.png")
                    else:
                        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_filename)
                else:
                    # Complete path or simple filename
                    if os.path.isabs(segmentation_image_url):
                        image_path = segmentation_image_url
                    else:
                        # Simple filename, build path in backend directory
                        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), segmentation_image_url)
                
                # Check if file exists
                if os.path.exists(image_path):
                    # Load and resize image
                    pil_image = PILImage.open(image_path)
                    
                    # Define maximum dimensions for image in report
                    max_width = 4.5 * inch
                    max_height = 3.5 * inch
                    
                    # Calculate aspect ratio
                    aspect_ratio = pil_image.width / pil_image.height
                    
                    if aspect_ratio > 1:  # Landscape
                        width = min(max_width, pil_image.width)
                        height = width / aspect_ratio
                    else:  # Portrait
                        height = min(max_height, pil_image.height)
                        width = height * aspect_ratio
                    
                    # Create ReportLab image
                    reportlab_image = Image(image_path, width=width, height=height)
                    reportlab_image.hAlign = 'CENTER'
                    
                    elements.append(reportlab_image)
                    
                    # Add image description
                    elements.append(Spacer(1, 10))
                    elements.append(Paragraph(
                        "3D segmentation image showing identified tumor contours. "
                        "Colored regions represent automatically segmented tumor areas.",
                        ParagraphStyle(
                            'ImageCaption',
                            parent=self.styles['Normal'],
                            fontSize=9,
                            textColor=HexColor('#4b5563'),
                            alignment=TA_CENTER,
                            fontName='Helvetica-Oblique'
                        )
                    ))
                    
                else:
                    elements.append(Paragraph(
                        f"Segmentation image not found at path: {image_path}",
                        self.styles['NormalText']
                    ))
                    
            except Exception as e:
                elements.append(Paragraph(
                    f"Error loading segmentation image: {str(e)}",
                    self.styles['NormalText']
                ))
        else:
            elements.append(Paragraph(
                "No segmentation image available for this analysis.",
                self.styles['NormalText']
            ))
        
        elements.append(Spacer(1, 15))
        return elements

    def _create_survival_section(self, analysis_data: Dict) -> List:
        """Create survival prediction section"""
        elements = []
        
        elements.append(Paragraph("SURVIVAL PREDICTION", self.styles['SectionHeader']))
        
        survival_pred = analysis_data.get('survival_prediction', {})
        
        if survival_pred:
            survival_days = survival_pred.get('survival_days', 0)
            risk_category = survival_pred.get('risk_category', 'Unknown')
            
            # Survival prediction table (without confidence)
            survival_data = [
                ["Prediction metric", "Value"],
                ["Estimated survival", f"{survival_days} days ({survival_days/30.44:.1f} months)"],
                ["Risk category", risk_category]
            ]
            
            survival_table = Table(survival_data, colWidths=[2.5*inch, 2.5*inch])
            survival_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('BACKGROUND', (0, 0), (0, -1), HexColor('#f3f4f6')),
            ]))
            
            elements.append(survival_table)
        else:
            elements.append(Paragraph(
                "No survival prediction data available.",
                self.styles['NormalText']
            ))
        
        elements.append(Spacer(1, 15))
        return elements

    def _create_features_section(self, analysis_data: Dict) -> List:
        """Create extracted features section"""
        elements = []
        
        elements.append(Paragraph("EXTRACTED RADIOMIC FEATURES", self.styles['SectionHeader']))
        
        features = analysis_data.get('extracted_features', {})
        
        if features:
            # Split features into two columns for better layout
            feature_items = list(features.items())
            mid_point = len(feature_items) // 2
            
            # Create features table
            features_data = [["Feature", "Value", "Feature", "Value"]]
            
            for i in range(max(mid_point, len(feature_items) - mid_point)):
                row = []
                
                # Left column
                if i < len(feature_items):
                    key, value = feature_items[i]
                    # Translate feature names to English
                    english_key = self._translate_feature_name(key)
                    row.extend([english_key, f"{value:.3f}"])
                else:
                    row.extend(["", ""])
                
                # Right column
                if i + mid_point < len(feature_items):
                    key, value = feature_items[i + mid_point]
                    english_key = self._translate_feature_name(key)
                    row.extend([english_key, f"{value:.3f}"])
                else:
                    row.extend(["", ""])
                
                features_data.append(row)
            
            features_table = Table(features_data, colWidths=[1.8*inch, 1*inch, 1.8*inch, 1*inch])
            features_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ]))
            
            elements.append(features_table)
        else:
            elements.append(Paragraph(
                "No radiomic features extracted.",
                self.styles['NormalText']
            ))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _translate_feature_name(self, key: str) -> str:
        """Translate feature names to English"""
        translations = {
            't1_3d_tumor_volume': '3D Tumor Volume',
            't1_3d_max_intensity': '3D Max Intensity',
            't1_3d_major_axis_length': '3D Major Axis Length',
            't1_3d_area': '3D Area',
            't1_3d_minor_axis_length': '3D Minor Axis Length',
            't1_3d_extent': '3D Extent',
            't1_3d_surface_to_volume_ratio': '3D Surface/Volume Ratio',
            't1_3d_glcm_contrast': '3D GLCM Contrast',
            't1_3d_mean_intensity': '3D Mean Intensity',
            't1_2d_area_median': '2D Median Area'
        }
        return translations.get(key, key.replace('_', ' ').title())

    def _create_recommendations_section(self, analysis_data: Dict) -> List:
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("CLINICAL RECOMMENDATIONS", self.styles['SectionHeader']))
        
        # Generate recommendations based on results
        recommendations = []
        
        # Check if there's a report text
        report_text = analysis_data.get('report', '')
        if report_text:
            recommendations.append(f"Analysis summary: {report_text}")
        
        # Add model-based recommendations
        if 'survival_prediction' in analysis_data:
            survival_pred = analysis_data['survival_prediction']
            risk_category = survival_pred.get('risk_category', '')
            
            if 'High Risk' in risk_category:
                recommendations.extend([
                    "Immediate multidisciplinary consultation recommended due to high-risk classification.",
                    "Consider aggressive treatment protocols and frequent monitoring.",
                    "Palliative care consultation may be beneficial."
                ])
            elif 'Medium Risk' in risk_category:
                recommendations.extend([
                    "Regular monitoring and standard treatment protocols recommended.",
                    "Oncological consultation for treatment planning.",
                    "Consider quality of life assessments."
                ])
            else:
                recommendations.extend([
                    "Standard follow-up protocols recommended.",
                    "Regular imaging studies to monitor progression.",
                    "Maintain current treatment regimen if applicable."
                ])
        
        # Add general recommendations
        recommendations.extend([
            "Complete histopathological examination for definitive diagnosis.",
            "Genetic testing may provide additional prognostic information.",
            "Patient and family counseling regarding diagnosis and prognosis.",
            "Coordination with neuro-oncology team for comprehensive care planning."
        ])
        
        for i, rec in enumerate(recommendations, 1):
            elements.append(Paragraph(
                f"{i}. {rec}",
                self.styles['NormalText']
            ))
        
        elements.append(Spacer(1, 15))
        return elements

    def _create_disclaimer(self, analysis_data: Dict) -> List:
        """Create medical disclaimer section"""
        elements = []
        
        elements.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", self.styles['SectionHeader']))
        
        disclaimer_text = """
        This report is generated by an AI-powered brain tumor analysis system for research and educational purposes only. 
        The results presented are computational predictions based on machine learning models and should NOT be considered 
        as definitive medical diagnoses or treatment recommendations.
        
        This analysis tool is NOT a substitute for professional medical diagnosis, treatment, or clinical judgment. 
        All results must be interpreted by qualified medical professionals in conjunction with complete clinical information, 
        patient history, and additional diagnostic tests.
        
        Survival predictions and risk stratifications are statistical estimates and may not reflect individual patient outcomes. 
        Treatment decisions should never be based solely on these AI-generated results.
        
        Segmentation results and radiomic features are provided for research purposes and require validation 
        by experienced radiologists and neuro-oncologists before any clinical application.
        
        URGENT: If this analysis suggests concerning results, please immediately consult with a 
        qualified neuro-oncologist or neurosurgeon for proper evaluation and treatment planning.
        """
        
        elements.append(Paragraph(disclaimer_text, self.styles['NormalText']))
        
        # Footer
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(
            "Generated by MedicImage Brain Tumor Analysis - Inveep Inc",
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