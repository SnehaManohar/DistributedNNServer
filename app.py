from flask import Flask, jsonify, request, render_template, Response

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
        model.get_queue().put_nowait(request.get_json())
        return ''
    return Response(status=400)


@app.route('/status')
def get_status():
    now, correct, total = model.accuracy
    return render_template('status.html',
                           accuracy='%.2f' % (correct * 100. / total),
                           date=now.strftime('%d-%m-%Y'),
                           time=now.strftime('%I:%M:%S%p'))


@app.route('/train')
def train():
    def train_progress():
        yield 'Request submitted<br/>'
        model.get_train_queue().put(None)
        yield 'Waiting for model<br/>'
        assert model.get_progress_queue().get() == 0
        yield 'Training<br/>'
        assert model.get_progress_queue().get() == 1
        yield 'Training complete<br/>'
    return Response(train_progress())


if __name__ == '__main__':
    app.run('0.0.0.0', 1729, threaded=True)
