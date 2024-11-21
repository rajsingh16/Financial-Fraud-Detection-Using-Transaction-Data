from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the Random Forest models
with open('rf_original.pkl', 'rb') as file:
    rf_original = pickle.load(file)

with open('rf_downsampled.pkl', 'rb') as file:
    rf_downsampled = pickle.load(file)

with open('rf_upsampled.pkl', 'rb') as file:
     rf_upsampled= pickle.load(file)

# Define the columns used in training
TRAINING_COLUMNS = ['User','Card','Year','Month','Day','Amount','Merchant Name','Zip','MCC']


# Function to preprocess the CSV data
def preprocess_data(df):
    # Clean the dataset
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)
    df['Is Fraud?'] = df['Is Fraud?'].map({'No': 0, 'Yes': 1})
    df = df.drop(columns=['Errors?'], errors='ignore')  # Dropping column with many missing values, if it exists
    df = df.dropna()  # Dropping rows with any missing values
    
    # Ensure all training columns are present
    for col in TRAINING_COLUMNS:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default values
    
    # Ensure only columns used during training are kept
    df = df[TRAINING_COLUMNS]
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        df = preprocess_data(df)
        
        # Make predictions
        predictions_original = rf_original.predict(df)
        predictions_downsampled = rf_downsampled.predict(df)
        predictions_upsampled = rf_upsampled.predict(df)

        # Map predictions: 0 -> Not Fraud, 1 -> Fraud
        def map_prediction(pred):
            return 'Not Fraud' if pred == 0 else 'Fraud'

        predictions_original_mapped = [map_prediction(pred) for pred in predictions_original]
        predictions_downsampled_mapped = [map_prediction(pred) for pred in predictions_downsampled]
        predictions_upsampled_mapped = [map_prediction(pred) for pred in predictions_upsampled]
        
        return jsonify({
            'predictions_original': predictions_original_mapped,
            'predictions_downsampled': predictions_downsampled_mapped,
            'predictions_upsampled': predictions_upsampled_mapped
        })


    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
