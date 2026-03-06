# Neonatal Health Prediction System

A Streamlit-based tool for predicting neonatal health risks using machine learning models.

## Features

- Predicts neonatal health risks based on 8 clinical features
- Beautiful user interface with responsive design
- Real-time prediction results display
- Detailed parameter descriptions and system information

## Input Features

1. Gestational Age (weeks)
2. Birth Weight (kg)
3. Head Circumference (cm)
4. Chest Circumference (cm)
5. Apgar Score (1 min)
6. Respiratory Distress Syndrome (Yes/No)
7. Invasive Mechanical Ventilation (days)
8. Non-invasive Mechanical Ventilation (days)

## Local Run

### Environment Requirements

- Python 3.10+
- Dependencies: see requirements.txt

### Installation Steps

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud

1. Push the code to a GitHub repository
2. Visit [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your GitHub repository
5. Set the main file path to `model1/app.py`
6. Click "Deploy"

## Project Structure

- `model1/app.py` - Streamlit application main file
- `model1/my_best_pipeline666.pkl` - Trained model file
- `model1/requirements.txt` - Dependencies file
- `model1/库的版本信息.txt` - Dependency version information

## Notes

- This system is only an auxiliary tool and cannot replace professional medical diagnosis
- The model is trained based on clinical data, and prediction results are for reference only
