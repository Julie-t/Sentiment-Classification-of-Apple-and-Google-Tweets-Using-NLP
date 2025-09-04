import pickle
from flask import Flask, request, jsonify, render_template

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    text = None

    # If the request is JSON
    if request.is_json:
        data = request.get_json()
        text = data.get("text")
    else:
        # If it's from an HTML form
        text = request.form.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Predict sentiment
    prediction = model.predict([text])[0]

    # If it came from the HTML form, render result on page
    if not request.is_json:
        return render_template("index.html", sentiment=prediction)

    # If it came from API (JSON), return JSON response
    return jsonify({"text": text, "sentiment": prediction})


if __name__ == "__main__":
    app.run(debug=True)