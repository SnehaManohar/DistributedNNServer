from flask import Flask, jsonify, request, render_template

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
    model.get_queue().put_nowait(request.get_json())
    return ''


@app.route('/status')
def get_status():
    now, correct, total = model.test()
    return render_template('status.html',
                           accuracy='%.2f' % (correct * 100. / total),
                           date=now.strftime('%d-%m-%Y'),
                           time=now.strftime('%I:%M:%S%p'))


@app.route('/train')
def train():
    model.train()
    return ''


if __name__ == '__main__':
    app.run('0.0.0.0', 1729)
