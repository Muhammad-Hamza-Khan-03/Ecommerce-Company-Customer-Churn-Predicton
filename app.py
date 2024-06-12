from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Define categorical columns for label encoding
cat_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract inputs from form
    inputs = {key: [value] for key, value in request.form.items()}
    
    # Create DataFrame from input
    input_df = pd.DataFrame(inputs)

    # Apply label encoding to categorical columns
    label_encoder = LabelEncoder()
    for col in cat_cols:
        input_df[col] = label_encoder.fit_transform(input_df[col])

    # Convert all columns to numeric (including categorical after label encoding)
    input_df = input_df.apply(pd.to_numeric)

    # Scale the data
    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(input_df)

    # Predict using the model
    prediction = model.predict(X_test_scaled)

    # Determine result based on prediction
    if prediction[0] == 1:
        res = "Churn"
    else:
        res = "Not Churn"

    # Pass the result to the template
    return render_template("index.html", prediction=res)

if __name__ == "__main__":
    app.run(debug=True)
