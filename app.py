from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_python_file')
def run_python_file():
    # Replace 'your_script.py' with the actual name of your Python file
    subprocess.run(["python", "predictions.py"])
    return "Python file executed successfully!"

if __name__ == '__main__':
    app.run(debug=True)




