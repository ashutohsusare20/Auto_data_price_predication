from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the model, scaler, and feature names
with open('car_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('features.pkl', 'rb') as features_file:
    all_features = pickle.load(features_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    data = {
        'Year': [float(request.form['year'])],
        'Engine HP': [float(request.form['engine_hp'])],
        'Engine Cylinders': [int(request.form['engine_cylinders'])],
        'Transmission Type_MANUAL': [1 if request.form['transmission'].lower() == 'manual' else 0],
        'Driven_Wheels_rear wheel drive': [1 if request.form['driven_wheels'].lower() == 'rear wheel drive' else 0],
        'Number of Doors': [int(request.form['number_of_doors'])],
        'Market Category_Factory Tuner,Luxury,High-Performance': [1 if 'factory tuner' in request.form['market_category'].lower() else 0],
        'Market Category_Luxury': [1 if 'luxury' in request.form['market_category'].lower() else 0],
        'Market Category_Performance': [1 if 'performance' in request.form['market_category'].lower() else 0],
        'Vehicle Size_Compact': [1 if request.form['vehicle_size'].lower() == 'compact' else 0],
        'Vehicle Style_Coupe': [1 if request.form['vehicle_style'].lower() == 'coupe' else 0],
        'Vehicle Style_Convertible': [1 if request.form['vehicle_style'].lower() == 'convertible' else 0],
        'highway MPG': [float(request.form['highway_mpg'])],
        'city mpg': [float(request.form['city_mpg'])],
        'Popularity': [float(request.form['popularity'])]
    }
    
    # Convert input data to DataFrame
    df = pd.DataFrame(data)
    
    # Reindex to match the model's expected feature set
    df_encoded = df.reindex(columns=all_features, fill_value=0)
    
    # Print the DataFrame to debug
    print("Encoded Features:\n", df_encoded)
    
    # Apply scaling
    final_input = scaler.transform(df_encoded)
    
    # Print the scaled input to debug
    print("Scaled Input:\n", final_input)
    
    # Predict
    prediction = model.predict(final_input)
    
    # Print the prediction to debug
    print("Prediction:\n", prediction)
    
    return render_template('index.html', prediction_text=f'Predicted Car Price: ${prediction[0]:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)
