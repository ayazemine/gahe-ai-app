"""
Prediction module for the Flask app
Handles loading trained models and making predictions
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class PredictionEngine:
    """Handles all prediction operations"""
    
    def __init__(self, experiment_dir="mrmr_experiment_results"):
        self.experiment_dir = experiment_dir
        self.scaler = None
        self.cnn_model = None
        self.mrmr_selector = None
        self.ml_models = {}  # Store all trained models
        self.experiment_info = None
        self.optimization_results = None
        self.best_model_name = None
        self.best_model = None
        
    def load_experiment(self):
        """Load all experiment data and models"""
        try:
            # Load experiment info
            with open(os.path.join(self.experiment_dir, 'experiment_info.pickle'), 'rb') as f:
                self.experiment_info = pickle.load(f)
            
            # Load optimization results
            self.optimization_results = pd.read_csv(
                os.path.join(self.experiment_dir, 'optimization_results.csv')
            )
            
            # Get best model
            best_idx = self.optimization_results['Avg_Test_R2'].idxmax()
            self.best_model_name = self.optimization_results.loc[best_idx, 'Model']
            
            # Load all available model pickles into ml_models
            for _, row in self.optimization_results.iterrows():
                model_name = row['Model']
                model_path = os.path.join(self.experiment_dir, f"model_{model_name}.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model_obj = pickle.load(f)
                        self.ml_models[model_name] = model_obj
                        if model_name == self.best_model_name:
                            self.best_model = model_obj
            
            # Load scaler
            scaler_path = os.path.join(self.experiment_dir, 'scaler.pickle')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                return False, "Scaler file not found (scaler.pickle)"

            # Load CNN feature extractor and mRMR selector
            cnn_path = os.path.join(self.experiment_dir, 'cnn_model.pickle')
            mrmr_path = os.path.join(self.experiment_dir, 'mrmr_selector.pickle')
            if os.path.exists(cnn_path):
                with open(cnn_path, 'rb') as f:
                    self.cnn_model = pickle.load(f)
            if os.path.exists(mrmr_path):
                with open(mrmr_path, 'rb') as f:
                    self.mrmr_selector = pickle.load(f)
            
            print(f"✅ Experiment loaded successfully")
            print(f"   Best model: {self.best_model_name}")
            print(f"   Loaded models: {list(self.ml_models.keys())}")
            print(f"   Original features: {self.experiment_info['n_features_original']}")
            print(f"   Selected features: {self.experiment_info['n_features_selected']}")
            
            return True, "Experiment loaded successfully"
            
        except FileNotFoundError:
            return False, "Experiment results not found"
        except Exception as e:
            return False, f"Error loading experiment: {str(e)}"
    
    def prepare_features(self, features_array):
        """Prepare and process features for prediction"""
        try:
            # Step 1: scale original inputs
            features_scaled = self.scaler.transform(features_array)

            # Step 2: CNN feature extraction (if available)
            if self.cnn_model is not None and hasattr(self.cnn_model, 'extract_features'):
                features_cnn = self.cnn_model.extract_features(features_scaled)
            else:
                features_cnn = features_scaled

            # Step 3: mRMR feature selection (if available)
            if self.mrmr_selector is not None and hasattr(self.mrmr_selector, 'transform'):
                features_processed = self.mrmr_selector.transform(features_cnn)
            else:
                features_processed = features_cnn
            
            return features_processed, True, "Features prepared successfully"
            
        except Exception as e:
            return None, False, f"Error preparing features: {str(e)}"
    
    def predict(self, features_array, model_name: str = None):
        """Make prediction with selected model (or best by default)"""
        try:
            # Choose model
            model_to_use = None
            chosen_model_name = None
            if model_name and model_name in self.ml_models:
                model_to_use = self.ml_models[model_name]
                chosen_model_name = model_name
            else:
                model_to_use = self.best_model
                chosen_model_name = self.best_model_name
            
            if model_to_use is None:
                return None, False, "Model not loaded"
            
            # Prepare features
            features_processed, success, message = self.prepare_features(features_array)
            if not success:
                return None, False, message
            
            # Make prediction
            prediction = model_to_use.predict(features_processed)[0]
            
            # Get confidence (example: based on feature range)
            confidence = 0.85
            
            return {
                'prediction': float(prediction),
                'confidence': confidence,
                'model': chosen_model_name,
                'features_info': {
                    'original': self.experiment_info['n_features_original'],
                    'cnn': self.experiment_info['n_features_cnn'],
                    'selected': self.experiment_info['n_features_selected']
                }
            }, True, "Prediction successful"
            
        except Exception as e:
            return None, False, f"Error making prediction: {str(e)}"
    
    def get_model_info(self):
        """Get information about all trained models"""
        try:
            if self.optimization_results is None:
                return None, False, "No optimization results available"
            
            info = []
            for _, row in self.optimization_results.iterrows():
                info.append({
                    'name': row['Model'],
                    'avg_test_r2': float(row['Avg_Test_R2']),
                    'std_test_r2': float(row['Std_Test_R2']),
                    'avg_test_mae': float(row['Avg_Test_MAE']),
                    'avg_test_rmse': float(row['Avg_Test_RMSE']),
                    'optimization_time': float(row['Optimization_Time'])
                })
            
            return info, True, "Model info retrieved"
            
        except Exception as e:
            return None, False, f"Error getting model info: {str(e)}"


# Global prediction engine instance
prediction_engine = None

def init_prediction_engine(experiment_dir="mrmr_experiment_results"):
    """Initialize the global prediction engine"""
    global prediction_engine
    prediction_engine = PredictionEngine(experiment_dir)
    return prediction_engine.load_experiment()

def get_prediction_engine():
    """Get the global prediction engine"""
    global prediction_engine
    if prediction_engine is None:
        init_prediction_engine()
    return prediction_engine
