from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("house_price_pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    input_df = pd.DataFrame([{
        "OverallQual": float(request.form["OverallQual"]),
        "GrLivArea": float(request.form["GrLivArea"]),
        "GarageCars": float(request.form["GarageCars"]),
        "GarageArea": float(request.form["GarageArea"]),
        "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
        "YearBuilt": float(request.form["YearBuilt"]),
        "FullBath": float(request.form["FullBath"]),
        "TotRmsAbvGrd": float(request.form["TotRmsAbvGrd"])
    }])

    prediction = model.predict(input_df)

    return render_template(
        "index.html",
        prediction=round(prediction[0], 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
