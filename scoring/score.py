import json
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import logging
import os
import base64

def init():
    global model, device, class_names
    
    logging.info("Starting initialization...")
    
    # Set device - use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    try:
        # Load your model
        model = models.resnet18(pretrained=False)
        num_classes = 6
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        # In Azure ML managed online endpoints, models are mounted to this path
        # No need to join with "models/model_2.pth" as that path is already specified in deployment
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model_2.pth")
        logging.info(f"Loading model from: {model_path}")
        
        # Add error handling for model loading
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}")
            # List files in the directory to debug
            model_dir = os.getenv("AZUREML_MODEL_DIR", "")
            if os.path.exists(model_dir):
                logging.info(f"Files in model directory: {os.listdir(model_dir)}")
            else:
                logging.error(f"Model directory {model_dir} does not exist")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Define class names
        class_names = ['Biryani', 'Milk Tea', 'Chapati', 'Chicken tikka', 'Paratha', 'Samosa'] 
        logging.info("Init Complete")
    except Exception as e:
        logging.error(f"Initialization error: {str(e)}")
        raise

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5773, 0.4623, 0.3385], 
                                std=[0.2559, 0.2411, 0.2455])
        ])
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

def run(raw_data):
    try:
        logging.info("Received inference request")
        
        # Parse input based on content type
        try:
            if isinstance(raw_data, str):
                # If it's a string, try to parse as JSON
                if not raw_data.strip():
                    logging.error("Input data is an empty string")
                    return {'Error': 'Input data is empty'}
                try:
                    input_data = json.loads(raw_data)
                except json.JSONDecodeError as e:
                    # If not valid JSON, assume it's base64 directly
                    logging.info("Input is not JSON, treating as direct base64")
                    image_base64 = raw_data
            else:
                # If already a dict or other object
                input_data = raw_data
                
            # Extract base64 data - handle both structured and direct formats
            if isinstance(input_data, dict):
                # Try to extract from structured format
                if 'input_data' in input_data and 'data' in input_data['input_data']:
                    image_base64 = input_data['input_data']['data']
                else:
                    # Try other common formats
                    image_base64 = input_data.get('data', input_data.get('image', ''))
            else:
                # Assume the input is directly the base64 string
                image_base64 = raw_data
                
            logging.info(f"Extracted base64 data of length: {len(str(image_base64))}")
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)
            logging.info(f"Decoded image bytes of length: {len(image_bytes)}")

            # Preprocess the image
            input_tensor = preprocess_image(image_bytes)  

            # Run inference
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get results
            predicted_idx = torch.argmax(probabilities).item()
            predicted_class = class_names[predicted_idx]
            confidence = probabilities[predicted_idx].item()
            
            # Return the prediction result
            result = {
                "predicted_class": predicted_class,
                "confidence": float(confidence)  # Convert to native Python float for JSON serialization
            }
            
            logging.info(f"Prediction result: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Error processing input data: {str(e)}")
            return {'Error': f'Input processing error: {str(e)}'}
    except Exception as e:
        error_message = {'Error': str(e)}
        logging.error(f"Error during inference: {error_message}")
        return error_message