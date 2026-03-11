from flask import Flask, render_template, request
import pandas as pd
import os
import sys

from src.transaction.pipelines.prediction_pipeline import PredictionPipeline
from src.transaction.logger import logger
from src.transaction.exception import CustomException

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            data = {
                "amt": float(request.form["amt"]),
                "age": int(request.form["age"]),
                "gender": request.form["gender"],
                "merchant": request.form["merchant"],
                "category": request.form["category"],
                "job": request.form["job"],
                "state": request.form["state"],
                "zip": int(request.form["zip"]),
                "lat": float(request.form["lat"]),
                "long": float(request.form["long"]),
                "merch_lat": float(request.form["merch_lat"]),
                "merch_long": float(request.form["merch_long"]),
                "city_pop": int(request.form["city_pop"]),
                "unix_time": int(request.form["unix_time"])
            }

            df = pd.DataFrame([data])

            predictor = PredictionPipeline()
            output = predictor.predict(df)

            result = {
                "probability": round(float(output["fraud_probability"][0]), 4),
                "prediction": int(output["prediction"][0])
            }

            logger.info("Prediction completed successfully")

        except Exception as e:
            logger.error("Prediction failed")
            error = "Prediction failed. Please check input values."

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
