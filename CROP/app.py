from flask import Flask, render_template, request
import joblib
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained models individually
models = {
    "eligible_pmfby": joblib.load("models/model_pmfby.pkl"),
    "eligible_pmkisan": joblib.load("models/model_pmkisan.pkl"),
    "eligible_fpo_support": joblib.load("models/model_fpo_support.pkl")
}

# Load label encoders
with open("models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load dataset to get dropdown options
df = pd.read_csv("Updated_Farmer_Dataset.csv")

# Feature list used for prediction
features = ['age', 'gender', 'education', 'land_size_acres', 'annual_income',
            'crop_type', 'state', 'district', 'region', 'has_taken_crop_insurance']

# ---------------------- Helper Functions ----------------------

def get_dropdown_options():
    return {
        "age_range": list(range(18, 81)),
        "genders": sorted(df['gender'].dropna().unique()),
        "educations": sorted(df['education'].fillna("Unknown").unique()),
        "crop_types": sorted(df['crop_type'].dropna().unique()),
        "states": sorted(df['state'].dropna().unique()),
        # "districts": sorted(df['district'].dropna().unique()),
        # "regions": sorted(df['region'].dropna().unique())
    }

def encode_input(input_data):
    """
    Encodes categorical inputs using label encoders and safely handles numeric values.
    Logs a warning for unseen categorical labels.
    """
    encoded = []
    for col in features:
        val = input_data[col]

        if col in label_encoders:
            le = label_encoders[col]
            if val in le.classes_:
                encoded_val = le.transform([val])[0]
            else:
                print(f"⚠️ Warning: unseen value '{val}' in column '{col}'")  # Log warning
                encoded_val = -1  # fallback for unseen value
        else:
            try:
                encoded_val = float(val)
            except ValueError:
                print(f"⚠️ Warning: invalid numeric value '{val}' in column '{col}', using default 0.0")
                encoded_val = 0.0  # fallback for malformed or missing numeric input

        encoded.append(encoded_val)

    return encoded


# ---------------------- Routes ----------------------

@app.route("/")
def index():
    options = get_dropdown_options()
    return render_template("form.html", **options)

@app.route("/get_districts_and_regions", methods=["POST"])
def get_districts_and_regions():
    selected_state = request.json.get("state")
    filtered = df[df["state"] == selected_state]
    districts = sorted(filtered["district"].dropna().unique())
    regions = sorted(filtered["region"].dropna().unique())
    return {"districts": districts, "regions": regions}


@app.route("/predict", methods=["POST"])
def predict():
    # Collect form inputs
    input_data = {
        "age": int(request.form["age"]),
        "gender": request.form["gender"],
        "education": request.form["education"],
        "land_size_acres": float(request.form["land_size_acres"]),
        "annual_income": float(request.form["annual_income"]),
        "crop_type": request.form["crop_type"],
        "state": request.form["state"],
        "district": request.form["district"],
        "region": request.form["region"],
        "has_taken_crop_insurance": int(request.form["has_taken_crop_insurance"])
    }

    # Label encode input data
    encoded_input = encode_input(input_data)

    # Run prediction for each model
    results = {}
    for key, model in models.items():
        prob = model.predict_proba([encoded_input])[0][1]  # probability of class 1
        results[key] = "Eligible" if prob >= 0.4 else "Not Eligible"
        # results[f"{key}_Probability"] = f"{prob:.2f}"
        # results[f"{key}_Prob_NotEligible"] = f"{1 - prob:.2f}"


    return render_template("result.html", results=results)

# ---------------------- Run App ----------------------

if __name__ == "__main__":
    app.run(debug=True)
