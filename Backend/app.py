import os
import base64
import io
import json
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from efficientnet_pytorch import EfficientNet
from report_generator import MedicalReportGenerator  # Import local sans Backend.
from datetime import datetime

app = FastAPI(title="MedicImage API", description="AI Skin Disease Classifier API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and device
model = None
device = None
class_names = ['Acne', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Eczemaa', 'Rosacea']
report_generator = MedicalReportGenerator()

# Pydantic models for request/response
class ClassificationRequest(BaseModel):
    image: str

class ClassificationResponse(BaseModel):
    success: bool
    predictions: Dict[str, float]
    primary_condition: str
    confidence: float
    class_names: list

class ReportRequest(BaseModel):
    image: str
    analysis_data: Dict[str, Any]
    patient_name: str = "Mr Ramzi Houidi"

class HealthResponse(BaseModel):
    status: str
    message: str

class ModelInfoResponse(BaseModel):
    model_type: str
    num_classes: int
    class_names: list
    device: str

def load_model():
    """Load the classification-based EfficientNet model"""
    global model, device
    
    # Set device (CPU for simplicity)
    device = torch.device("cpu")
    
    # Load classification model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, len(class_names))
    
    classifier_path = os.path.join(os.path.dirname(__file__), '..', 'ML', 'DermaScan', 'disease_classifier.pth')
    if os.path.exists(classifier_path):
        model.load_state_dict(torch.load(classifier_path, map_location=device))
        print(f"Loaded classification model from {classifier_path}")
    else:
        raise FileNotFoundError(f"Classification model not found at {classifier_path}")
    
    model.to(device)
    model.eval()
    print("Classification model loaded successfully!")

def preprocess_image(image_data: str) -> torch.Tensor:
    """Preprocess the image to match the training pipeline"""
    # Decode base64 image
    if image_data.startswith('data:image'):
        # Remove data URL prefix
        image_data = image_data.split(',')[1]
    
    # Decode base64 to bytes
    image_bytes = base64.b64decode(image_data)
    
    # Open image with PIL
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Apply the same transformations as in the notebook
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    print("Loading skin disease classifier model...")
    load_model()

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", 
        message="Skin Disease Classifier API is running"
    )

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_skin_disease(request: ClassificationRequest):
    """Classify skin diseases from uploaded image"""
    try:
        # Preprocess the image
        image_tensor = preprocess_image(request.image)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        # Create results dictionary
        results = {}
        for i, class_name in enumerate(class_names):
            results[class_name] = float(probabilities[i])
        
        # Find primary condition (highest probability)
        primary_condition = class_names[np.argmax(probabilities)]
        confidence = float(np.max(probabilities))
        
        return ClassificationResponse(
            success=True,
            predictions=results,
            primary_condition=primary_condition,
            confidence=confidence,
            class_names=class_names
        )
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/api/generate-report")
async def generate_medical_report(request: ReportRequest):
    """Generate a medical report PDF"""
    try:
        # Generate the PDF report
        pdf_bytes = report_generator.create_report(
            analysis_data=request.analysis_data,
            image_data=request.image,
            patient_name=request.patient_name
        )
        
        # Return the PDF as a downloadable file
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=medical_report_{request.patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.post("/api/generate-pdf-report")
async def generate_pdf_medical_report(request: ReportRequest):
    """Generate a PDF medical report - unified endpoint name"""
    try:
        # Generate the PDF report
        pdf_bytes = report_generator.create_report(
            analysis_data=request.analysis_data,
            image_data=request.image,
            patient_name=request.patient_name
        )
        
        # Return the PDF as a downloadable file
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=dermascan_report_{request.patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
        
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF report generation failed: {str(e)}")

@app.get("/api/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    return ModelInfoResponse(
        model_type="Classification Model",
        num_classes=len(class_names),
        class_names=class_names,
        device=str(device)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)  # Port 5002 pour éviter conflit 