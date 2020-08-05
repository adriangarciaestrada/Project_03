from flask import Flask, render_template, request, redirect
import run_model

app = Flask(__name__)

# Use flask_pymongo to set up mongo connection

text_prediction = ""
comments = ""

@app.route("/")
def index():
    return render_template("index.html", result = text_prediction)

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        comments = request.form['comments']
        print(comments)
        text_prediction = run_model.run_model_text(comments)
        print(text_prediction)
    return render_template("index.html", result = text_prediction, comment = comments)

if __name__ == "__main__":
    app.run(debug=True)