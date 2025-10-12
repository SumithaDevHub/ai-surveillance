from flask import Flask, render_template

app = Flask(__name__)

# Landing page
@app.route("/")
def index():
    return render_template("index.html")

# Dashboard page
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")  # Make a dashboard.html template

if __name__ == "__main__":
    app.run(debug=True)
