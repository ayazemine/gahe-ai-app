GAHE Outlet Air Temperature Prediction
========================================

A web-based prediction interface for Ground-Air Heat Exchanger (GAHE) outlet air temperature. The system uses a pre-trained hybrid pipeline: **CNN feature extraction → mRMR feature selection → GWO-optimized ML regression**.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python app.py
```

### 3. Open in browser

Navigate to **http://127.0.0.1:5000**

## Available Models

Four ML models are pre-trained with GWO-optimized hyperparameters:

- **CatBoost**
- **XGBoost**
- **LightGBM**
- **AdaBoost**

The user can select any model from the web interface dropdown.
## Dataset

### dataset_new.csv

This repository includes an openly accessible dataset:

**File:** `dataset_new.csv`  

**Description:**  
The dataset contains **532 experimental data points** obtained from a Ground-Air Heat Exchanger (GAHE) system. Each record consists of the input parameters used in the prediction model and the corresponding outlet air temperature values.

### Input Parameters

| Symbol | Parameter | Unit | Description |
|--------|-----------|------|-------------|
| T_inlet | Air Inlet Temperature | K | Inlet air temperature of the GAHE system |
| T_soil | Soil Temperature | K | Underground soil temperature |
| Re | Reynolds Number | — | Flow regime indicator |
| D | Pipe Diameter | m | Diameter of the buried pipe |
| H | Burial Depth | m | Depth of the pipe below ground |
| L | Pipe Length | m | Total length of the buried pipe |

### Data Availability and Usage Rights

The dataset has been made **openly accessible by the authors** for academic and research purposes.

- The dataset may be freely used in **academic and scientific studies**.  
- Proper citation of the related published article is required when using this dataset in any publication.  

If you use this dataset in your research, please ensure that you cite the associated publication.

## File Structure

```
GAHE-AI/
├── app.py                          # Flask application (entry point)
├── prediction_engine.py            # Loads trained models & makes predictions
├── hybrid_model.py                 # CNN feature extractor architecture
├── mRMR.py                         # mRMR feature selection
├── dataset_new.csv                 # Open-access Dataset
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Web interface
└── mrmr_experiment_results/        # Pre-trained model artifacts
    ├── experiment_info.pickle       # Experiment metadata
    ├── scaler.pickle                # Fitted StandardScaler
    ├── cnn_model.pickle             # Trained CNN feature extractor
    ├── mrmr_selector.pickle         # Trained mRMR selector
    ├── model_CatBoost.pkl           # Trained CatBoost model
    ├── model_XGBoost.pkl            # Trained XGBoost model
    ├── model_LightGBM.pkl           # Trained LightGBM model
    ├── model_AdaBoost.pkl           # Trained AdaBoost model
    └── optimization_results.csv     # Model performance summary
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model not loaded" | Ensure all `.pkl` and `.pickle` files exist in `mrmr_experiment_results/` |
| Import error | Run `pip install -r requirements.txt` |
| Port in use | Change the port in `app.py` |


## License

This project was created for academic research purposes.
