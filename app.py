from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the ML model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

# Crop Dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Ideal Environment for Crops (Example Values)
crop_environment = {
    "Rice": {"Nitrogen": 90, "Phosphorus": 40, "Potassium": 40, "Temperature": 25, "Humidity": 80, "pH": 6.5, "Rainfall": 200},
    "Maize": {"Nitrogen": 80, "Phosphorus": 50, "Potassium": 50, "Temperature": 22, "Humidity": 75, "pH": 6.0, "Rainfall": 150},
    "Apple": {"Nitrogen": 50, "Phosphorus": 30, "Potassium": 60, "Temperature": 18, "Humidity": 60, "pH": 6.8, "Rainfall": 120},
    "Banana": {"Nitrogen": 100, "Phosphorus": 50, "Potassium": 60, "Temperature": 27, "Humidity": 85, "pH": 6.0, "Rainfall": 250},
    "Mango": {"Nitrogen": 70, "Phosphorus": 40, "Potassium": 50, "Temperature": 30, "Humidity": 70, "pH": 6.5, "Rainfall": 100},
    # Add all crops with their ideal conditions
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get user input
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    # Preprocess input for prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    
    # Make prediction
    prediction = model.predict(sc_mx_features)

    # Get crop name from prediction
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated right now."
    else:
        result = "Sorry, we could not determine the best crop for the provided data."

    return render_template('index.html', result=result)

@app.route("/get_environment", methods=['POST'])
def get_environment():
    crop_name = request.form['crop_name'].capitalize()

    if crop_name in crop_environment:
        env = crop_environment[crop_name]
        environment_result = (f"Ideal conditions for {crop_name}: "
                              f"Nitrogen: {env['Nitrogen']}, Phosphorus: {env['Phosphorus']}, Potassium: {env['Potassium']}, "
                              f"Temperature: {env['Temperature']}°C, Humidity: {env['Humidity']}%, pH: {env['pH']}, "
                              f"Rainfall: {env['Rainfall']}mm")
    else:
        environment_result = "Crop not found. Please enter a valid crop name."

    return render_template("index.html", environment_result=environment_result)

if __name__ == "__main__":
    app.run(debug=True,port=5000)
