import os
import io
import json
import uvicorn
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from enum import Enum
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

# Define the skin conditions based on your notebook
SKIN_CONDITIONS = [
    "ACK-Actinic Keratosis",
    "BCC-Basal Cell Carcinoma", 
    "MEL-Melanoma", 
    "NEV-Nevus", 
    "SCC-Squamous Cell Carcinoma", 
    "SEK-Seborrheic Keratosis"
]  # Update this list with your actual skin conditions

# Define the ConvNeXt model with metadata
class ConvNeXtWithMetadata(nn.Module):
    def __init__(self, num_classes, metadata_features=5, convnext_variant='tiny'):
        super(ConvNeXtWithMetadata, self).__init__()
        
        # Import dynamically here to avoid loading at module level
        import torchvision.models as models
        
        # Load the appropriate ConvNeXt variant
        if convnext_variant == 'tiny':
            self.base_model = models.convnext_tiny(pretrained=True)
            num_features = 768  # Feature dimension for convnext_tiny
        elif convnext_variant == 'small':
            self.base_model = models.convnext_small(pretrained=True)
            num_features = 768  # Feature dimension for convnext_small
        elif convnext_variant == 'base':
            self.base_model = models.convnext_base(pretrained=True)
            num_features = 1024  # Feature dimension for convnext_base
        elif convnext_variant == 'large':
            self.base_model = models.convnext_large(pretrained=True)
            num_features = 1536  # Feature dimension for convnext_large
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {convnext_variant}")
        
        # Modify the classifier to output features instead of classification
        self.base_model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Metadata processing layers
        self.metadata_layers = nn.Sequential(
            nn.Linear(metadata_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined layers
        self.combined_layers = nn.Sequential(
            nn.Linear(num_features + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, metadata):
        # Process images through ConvNeXt
        img_features = self.base_model(images)
        
        # Process metadata
        metadata_features = self.metadata_layers(metadata)
        
        # Combine features
        combined = torch.cat((img_features, metadata_features), dim=1)
        
        # Final classification
        output = self.combined_layers(combined)
        
        return output

# Define input models
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"

class YesNo(str, Enum):
    YES = "yes"
    NO = "no"

class PatientMetadata(BaseModel):
    gender: Gender
    age: int
    smoke: YesNo
    drink: YesNo
    skin_cancer_history: YesNo

class PredictionResponse(BaseModel):
    predicted_condition: str
    confidence: float
    all_probabilities: dict

# Initialize FastAPI app
app = FastAPI(
    title="Skin Cancer Detection API",
    description="API for detecting skin cancer from images and patient metadata",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model
model = None
device = None
transform = None

@app.on_event("startup")
async def load_model():
    global model, device, transform
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model_path = "model/final_convnext_tiny.pth"
    
    # Check if model exists, if not you would need to provide it
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please download the model.")
        # In production, you might want to download the model from a storage service
    
    # Initialize model
    try:
        model = ConvNeXtWithMetadata(
            num_classes=len(SKIN_CONDITIONS),
            metadata_features=5,
            convnext_variant='tiny'
        ).to(device)
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully")
        
        # Define transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    except Exception as e:
        print(f"Error loading model: {e}")

def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    """Save an upload file to a temporary file"""
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
        upload_file.file.close()
        return tmp_path
    except Exception:
        return None

def predict_skin_condition(image_path, metadata):
    """
    Make prediction using the trained model
    
    Args:
        image_path: Path to the image file
        metadata: Dictionary containing patient metadata
    
    Returns:
        predicted_class: The predicted skin condition
        probabilities: Probabilities for each class
    """
    global model, device, transform
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    
    # Process metadata
    try:
        # Convert string values to appropriate numeric values
        smoke = 1 if str(metadata.get('smoke', '')).lower() in ['yes', 'true', '1', 'y', 't'] else 0
        drink = 1 if str(metadata.get('drink', '')).lower() in ['yes', 'true', '1', 'y', 't'] else 0
        history = 1 if str(metadata.get('skin_cancer_history', '')).lower() in ['yes', 'true', '1', 'y', 't'] else 0
        
        # Process age
        try:
            age = float(metadata.get('age', 50.0))
        except (ValueError, TypeError):
            age = 50.0  # Default age if invalid
        age_normalized = age / 100.0  # Normalize age
        
        # Process gender
        if isinstance(metadata.get('gender'), (int, float)):
            gender = float(metadata.get('gender'))
        elif isinstance(metadata.get('gender'), str):
            gender = 1 if metadata.get('gender', '').lower() in ['male', 'm', '1'] else 0
        else:
            gender = 0  # Default gender
        
        # Create metadata tensor
        metadata_tensor = torch.tensor([
            smoke,
            drink,
            history,
            age_normalized,
            gender
        ], dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing metadata: {str(e)}")
    
    # Get prediction
    try:
        with torch.no_grad():
            outputs = model(image_tensor, metadata_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Get the predicted class name
        predicted_class = SKIN_CONDITIONS[predicted.item()]
        probabilities_list = probabilities[0].cpu().numpy()
        
        # Create a dictionary of probabilities for each class
        class_probabilities = {SKIN_CONDITIONS[i]: float(probabilities_list[i]) for i in range(len(SKIN_CONDITIONS))}
        
        return predicted_class, class_probabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict-final", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    gender: Gender = Form(...),
    age: int = Form(...),
    smoke: YesNo = Form(...),
    drink: YesNo = Form(...),
    skin_cancer_history: YesNo = Form(...)
):
    """
    Predict skin cancer condition from an image and patient metadata.
    
    - **image**: The skin lesion image file
    - **gender**: Patient's gender (male/female)
    - **age**: Patient's age
    - **smoke**: Whether the patient smokes (yes/no)
    - **drink**: Whether the patient drinks alcohol (yes/no)
    - **skin_cancer_history**: Whether the patient has skin cancer history (yes/no)
    
    Returns the predicted skin condition and confidence scores.
    """
    # Validate age
    if age <= 0 or age > 120:
        raise HTTPException(status_code=400, detail="Age must be between 1 and 120")
    
    # Create metadata dictionary
    metadata = {
        "gender": gender,
        "age": age,
        "smoke": smoke,
        "drink": drink,
        "skin_cancer_history": skin_cancer_history
    }
    
    # Save uploaded image to temporary file
    temp_file = save_upload_file_tmp(image)
    if temp_file is None:
        raise HTTPException(status_code=400, detail="Failed to process the image")
    
    try:
        # Get prediction
        predicted_class, class_probabilities = predict_skin_condition(temp_file, metadata)
        
        # Get confidence score for predicted class
        confidence = class_probabilities[predicted_class]
        
        # Delete temporary file
        os.unlink(temp_file)
        
        # Return response
        return PredictionResponse(
            predicted_condition=predicted_class,
            confidence=confidence,
            all_probabilities=class_probabilities
        )
    except Exception as e:
        # Clean up temp file in case of error
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "Service is up and running"}

@app.get("/conditions")
async def get_conditions():
    """Get the list of skin conditions the model can detect"""
    return {"conditions": SKIN_CONDITIONS}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
