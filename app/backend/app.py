#=============================================================================
# AI vs Human Image Detector - Flask Backend Application
# 
# This application provides the backend API for the AI vs Human detector, with:
# - Multiple model support (Vision Transformer, ResNet, Ensemble)
# - RESTful API endpoints for model information and predictions
# - Confidence scoring for predictions
#
# The application uses PyTorch and timm for model handling, and Flask
# for API exposure. It's designed for salability by allowing new models to be added simply via models_config.json.
#============================================================================
import os
import json
import torch
import timm
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as tv_models
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_CONFIG_PATH'] = 'models_config.json'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

available_timm_models = list(timm.list_models())

def load_model_config():
    config_path = app.config['MODEL_CONFIG_PATH']
    
    if not os.path.exists(config_path):
        print(f"Model configuration file not found at {config_path}, creating default config")
        default_config = {
            "models": [
                {
                    "id": "vit",
                    "type": "vit",
                    "name": "Vision Transformer (ViT)",
                    "arch": "vit_base_patch16_224",
                    "model_path": "models/ai_vs_human_vit_model.pth",
                    "description": "Base model with global attention mechanism"
                },
                {
                    "id": "resnet",
                    "type": "resnet",
                    "name": "ResNet 50",
                    "arch": "resnet50",
                    "model_path": "models/ai_vs_human_resnet_model.pth",
                    "description": "CNN model with residual connections"
                }
            ],
            "ensemble": {
                "id": "ensemble",
                "name": "Ensemble Model",
                "model_path": "models/ai_vs_human_ensemble_model.pth",
                "description": "Combined predictions from Vision Transformer and ResNet models"
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading model configuration: {e}")
        return {"models": [], "ensemble": None}

model_config = load_model_config()

MODEL_CONFIGS = {}
for model in model_config.get("models", []):
    MODEL_CONFIGS[model["id"]] = model

ENSEMBLE_CONFIG = model_config.get("ensemble", {})
ENSEMBLE_MODEL_PATH = ENSEMBLE_CONFIG.get("model_path", "models/ai_vs_human_ensemble_model.pth")

def create_vit_model(model_arch, pretrained=False, dropout_rate=0.2):
    model = timm.create_model(model_arch, pretrained=pretrained, drop_rate=dropout_rate)
    
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, 1)
    
    return model

def create_resnet_model(pretrained=False, dropout_rate=0.2):
    if pretrained:
        try:
            model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        except:
            try:
                model = tv_models.resnet50(weights="IMAGENET1K_V2")
            except:
                model = tv_models.resnet50(pretrained=True)
    else:
        try:
            model = tv_models.resnet50(weights=None)
        except:
            model = tv_models.resnet50(pretrained=False)
        
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(dropout_rate),
        nn.Linear(512, 1)
    )
    
    return model

class AIvsHumanEnsemble(nn.Module):
    def __init__(self, vit_model=None, resnet_model=None):
        super(AIvsHumanEnsemble, self).__init__()
        self.vit_model = vit_model
        self.resnet_model = resnet_model
        
    def forward(self, x):
        vit_output = self.vit_model(x)
        resnet_output = self.resnet_model(x)
        
        ensemble_output = (vit_output + resnet_output) / 2.0
        
        return ensemble_output

def is_model_available(model_id):
    if model_id in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_id]
        model_type = config.get('type', '').lower()
        model_path = config.get('model_path', '')
        
        if not os.path.exists(model_path):
            return False
            
        if model_type == 'vit':
            arch = config.get('arch', '')
            return arch in available_timm_models
        elif model_type == 'resnet':
            return True
        else:
            arch = config.get('arch', '')
            return arch in available_timm_models
    
    elif model_id == 'ensemble':
        vit_available = any(cfg.get('type') == 'vit' and is_model_available(cfg_id) 
                            for cfg_id, cfg in MODEL_CONFIGS.items())
        resnet_available = any(cfg.get('type') == 'resnet' and is_model_available(cfg_id)
                              for cfg_id, cfg in MODEL_CONFIGS.items())
        return vit_available and resnet_available
    
    return False

def load_model(model_id):
    try:
        if model_id == 'ensemble':
            vit_model = None
            resnet_model = None
            
            for cfg_id, config in MODEL_CONFIGS.items():
                if config.get('type') == 'vit' and vit_model is None:
                    vit_model = load_model(cfg_id)
                    if vit_model is None:
                        print("Failed to load ViT model for ensemble")
                        return None
                        
                elif config.get('type') == 'resnet' and resnet_model is None:
                    resnet_model = load_model(cfg_id)
                    if resnet_model is None:
                        print("Failed to load ResNet model for ensemble")
                        return None
            
            if vit_model is None or resnet_model is None:
                print("Cannot create ensemble without both ViT and ResNet models")
                return None
            
            model = AIvsHumanEnsemble(vit_model, resnet_model).to(device)
            model.eval()
            
            print("Ensemble model created successfully!")
            return model

        if model_id not in MODEL_CONFIGS:
            print(f"Unknown model ID: {model_id}")
            return None
        
        config = MODEL_CONFIGS[model_id]
        model_type = config.get('type', '').lower()
        model_path = config.get('model_path', '')
        
        if model_type == 'vit':
            model_arch = config.get('arch', '')
            if model_arch not in available_timm_models:
                print(f"Architecture {model_arch} not available in timm")
                return None
                
            model = create_vit_model(model_arch, pretrained=False).to(device)
        elif model_type == 'resnet':
            model = create_resnet_model(pretrained=False).to(device)
            print("Created base ResNet50 model successfully")
        else:
            print(f"Unsupported model type: {model_type}")
            return None
        
        if not os.path.exists(model_path):
            print(f"Model weights not found at {model_path}")
            return None
            
        try:
            print(f"Loading weights from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            
            if isinstance(state_dict, dict):
                print(f"State dict keys: {list(state_dict.keys())[:5]}...")
                print(f"Total keys: {len(state_dict)}")
            else:
                print(f"State dict is not a dictionary, type: {type(state_dict)}")
            
            model.load_state_dict(state_dict)
            print(f"Model {model_id} loaded successfully!")
        except Exception as e:
            print(f"Error loading model state dict: {e}")
            print("Attempting alternative loading approaches...")
            
            try:
                if isinstance(state_dict, dict):
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if not k.startswith('model.'):
                            new_state_dict[f'model.{k}'] = v
                        else:
                            new_state_dict[k] = v
                    
                    try:
                        print("Trying with 'model.' prefix added")
                        model.load_state_dict(new_state_dict)
                        print(f"Model {model_id} loaded with prefixed state dict!")
                    except Exception as e1:
                        print(f"Failed with prefixed state dict: {e1}")
                        
                        if all(k.startswith('model.') for k in state_dict.keys()):
                            new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                            try:
                                print("Trying with 'model.' prefix removed")
                                model.load_state_dict(new_state_dict)
                                print(f"Model {model_id} loaded with stripped state dict!")
                            except Exception as e2:
                                print(f"Failed with stripped state dict: {e2}")
                                return None
                        else:
                            return None
                else:
                    print("State dict is not a dictionary, cannot attempt fixes")
                    return None
            except Exception as e2:
                print(f"All loading attempts failed: {e2}")
                import traceback
                traceback.print_exc()
                return None
                
        model.eval()
        return model
    except Exception as e:
        print(f"Unexpected error loading model {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_with_model(image_path, model, model_id):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probability = torch.sigmoid(output).item()
            
            result = {
                'prediction': 'AI-Generated' if probability >= 0.5 else 'Human-Created',
                'probability_ai': float(probability),
                'success': True
            }
            
            return result
    except Exception as e:
        print(f"Error predicting with model {model_id}: {e}")
        return {"error": str(e), "success": False}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Available models: {', '.join(loaded_models.keys())}")
        model_results = {}
        
        available_models = {model_id: model for model_id, model in loaded_models.items() if model is not None}
        
        if not available_models:
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'error': 'No models available for prediction'}), 500
        
        model_id = request.form.get('model_id', None)
        if model_id and model_id in available_models:
            try:
                result = predict_with_model(filepath, available_models[model_id], model_id)
                model_results[model_id] = result
            except Exception as e:
                print(f"Error running prediction with {model_id}: {e}")
                model_results[model_id] = {"error": str(e), "success": False}
        else:
            for m_id, model in available_models.items():
                if m_id != 'ensemble':
                    try:
                        result = predict_with_model(filepath, model, m_id)
                        model_results[m_id] = result
                    except Exception as e:
                        print(f"Error running prediction with {m_id}: {e}")
                        model_results[m_id] = {"error": str(e), "success": False}
        
        ensemble_result = None
        if (not model_id or model_id == 'ensemble') and 'ensemble' in available_models:
            try:
                ensemble_result = predict_with_model(filepath, available_models['ensemble'], 'ensemble')
            except Exception as e:
                print(f"Error running ensemble prediction: {e}")
        
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Error removing file: {e}")
        
        successful_results = {k: v for k, v in model_results.items() if v and v.get('success', True)}
        
        if not successful_results and (not ensemble_result or not ensemble_result.get('success', False)):
            return jsonify({'error': 'All models failed to process the image'}), 500
        
        return jsonify({
            'model_results': successful_results,
            'ensemble_result': ensemble_result if ensemble_result and ensemble_result.get('success', False) else None
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/models', methods=['GET'])
def get_models():
    model_list = []
    
    for model_id, config in MODEL_CONFIGS.items():
        is_available = is_model_available(model_id)
        model_info = {
            'id': model_id,
            'name': config.get('name', model_id),
            'description': config.get('description', ''),
            'available': is_available
        }
        model_list.append(model_info)
    
    if is_model_available('ensemble'):
        model_list.append({
            'id': 'ensemble',
            'name': ENSEMBLE_CONFIG.get('name', 'Ensemble Model'),
            'description': ENSEMBLE_CONFIG.get('description', 'Combined predictions from all models'),
            'available': True
        })
    
    return jsonify(model_list)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("Pre-loading models...")
    loaded_models = {}
    
    for model_id in MODEL_CONFIGS:
        try:
            model = load_model(model_id)
            if model is not None:
                loaded_models[model_id] = model
                print(f"Successfully loaded model: {model_id}")
            else:
                print(f"Model {model_id} could not be loaded")
        except Exception as e:
            print(f"Could not load model {model_id}: {e}")
    
    vit_model = None
    resnet_model = None
    
    for model_id, model in loaded_models.items():
        config = MODEL_CONFIGS.get(model_id, {})
        model_type = config.get('type', '').lower()
        
        if model_type == 'vit' and vit_model is None:
            vit_model = model
        elif model_type == 'resnet' and resnet_model is None:
            resnet_model = model
    
    if vit_model is not None and resnet_model is not None:
        try:
            ensemble_model = AIvsHumanEnsemble(vit_model, resnet_model).to(device)
            ensemble_model.eval()
            loaded_models['ensemble'] = ensemble_model
            print("Ensemble model created successfully")
        except Exception as e:
            print(f"Could not create ensemble model: {e}")
    else:
        print("Not creating ensemble model since we don't have both ViT and ResNet models")
    
    if not loaded_models:
        print("WARNING: No models were successfully loaded!")
    else:
        print(f"Successfully loaded {len(loaded_models)} models: {', '.join(loaded_models.keys())}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)