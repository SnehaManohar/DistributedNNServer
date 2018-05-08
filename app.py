from flask import Flask, jsonify, request, render_template, send_file

from model import Model

app = Flask('Sonder')
model = Model()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/model', methods=['GET'])
def fetch_model():
    return jsonify(model.get_parameters())


@app.route('/model', methods=['POST'])
def update_model():
    parameters = request.get_json()['params']
    if model.check_parameters(parameters):
        model.merge(parameters)
        return jsonify({'status': 'OK'})
    return '', 400


@app.route('/status')
def get_status():
    correct, total = model.accuracy
    return render_template('status.html', correct=correct, total=total, accuracy=round(correct * 100 / total))


@app.route('/apk')
def apk():
    return send_file('static/Sonder.apk', as_attachment=True, attachment_filename='Sonder.apk')


if __name__ == '__main__':
    app.run('0.0.0.0', 1729, threaded=True)
