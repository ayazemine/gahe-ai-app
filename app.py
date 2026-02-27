import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings('ignore')

# Import models and selectors
from prediction_engine import PredictionEngine

REQUIRED_FIELDS = ['havaGiris', 'toprak', 're', 'D', 'H', 'L']

def validate_input(data):
    """Validate prediction input data."""
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing field: {field}"
        try:
            float(data[field])
        except (ValueError, TypeError):
            return False, f"{field}: Invalid number format"
    return True, "Input is valid"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Global prediction engine
prediction_engine = None
experiment_dir = "mrmr_experiment_results"

def init_app():
    """Initialize the app and load models"""
    global prediction_engine
    prediction_engine = PredictionEngine(experiment_dir)
    success, message = prediction_engine.load_experiment()
    if success:
        print(f"✅ {message}")
    else:
        print(f"⚠️ {message}")
    return success

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction endpoint"""
    try:
        if prediction_engine is None or prediction_engine.best_model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first.'
            }), 503
        
        data = request.get_json()
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Error: {message}'
            }), 400
        
        # Create feature array
        features = np.array([
            float(data['havaGiris']),
            float(data['toprak']),
            float(data['re']),
            float(data['D']),
            float(data['H']),
            float(data['L'])
        ]).reshape(1, -1)
        
        # Optional model selection from request
        selected_model = None
        if isinstance(data, dict):
            selected_model = data.get('model')

        # Make prediction
        result, success, message = prediction_engine.predict(features, model_name=selected_model)
        
        if not success:
            return jsonify({
                'success': False,
                'error': message
            }), 500
        
        # Format response
        prediction_response = {
            'success': True,
            'input': {
                'havaGiris': float(data['havaGiris']),
                'toprak': float(data['toprak']),
                're': float(data['re']),
                'D': float(data['D']),
                'H': float(data['H']),
                'L': float(data['L'])
            },
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'model': result['model'],
            'features_count': {
                'original': result['features_info']['original'],
                'after_cnn': result['features_info']['cnn'],
                'after_mrmr': result['features_info']['selected']
            },
            'message': f'Prediction made with {result["model"]} model'
        }
        
        return jsonify(prediction_response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/api/info', methods=['GET'])
def get_info():
    """Get experiment info and model status"""
    try:
        info = {
            'models_trained': os.path.exists(experiment_dir),
            'experiment_dir': experiment_dir,
            'models_loaded': prediction_engine is not None and prediction_engine.best_model is not None
        }
        
        if prediction_engine and prediction_engine.experiment_info:
            info['experiment_info'] = {
                'n_features_original': prediction_engine.experiment_info.get('n_features_original', 0),
                'n_features_cnn': prediction_engine.experiment_info.get('n_features_cnn', 0),
                'n_features_selected': prediction_engine.experiment_info.get('n_features_selected', 0),
                'n_samples_total': prediction_engine.experiment_info.get('n_samples_total', 0),
                'experiment_date': prediction_engine.experiment_info.get('experiment_date', 'N/A')
            }
            
            # Get model info
            model_info, success, _ = prediction_engine.get_model_info()
            if success:
                info['models'] = model_info
        
        return jsonify(info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize app
    init_app()
    app.run(debug=True, host='127.0.0.1', port=5000)
