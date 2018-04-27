import json
from queue import Queue
from threading import Lock, Timer, Thread

import utils
from net import NeuralNet
from parameters import Parameters as P


class Model:
    def __init__(self):
        self.queue = Queue()
        with open('config.json') as config_file:
            self.config = json.load(config_file)
        self.model = NeuralNet(self.config)
        with open('params.json') as f:
            self.load_params(json.load(f))
        self.params = self.config['optimizer']['parameters']
        self.params['params'] = self.model.parameters()
        self.parameter_lengths = list(map(lambda x: x.data.nelement(), self.model.parameters()))
        self.optimizer = utils.create_object(**self.config['optimizer'])
        self.accuracy = None
        self.lock = Lock()
        self.changed = True
        self.params_updated = False

        self.loader = utils.get_data()

        self.convert_params()
        self.compute_accuracy()
        Thread(target=self.update_model, daemon=True).start()

    def get_parameters(self):
        return self.config

    def check_parameters(self, parameters):
        try:
            return (len(parameters) == len(self.parameter_lengths) and
                    all(map(lambda x, y: len(x) == y, parameters, self.parameter_lengths)))
        except TypeError:
            return False

    def convert_params(self):
        if self.changed:
            with self.lock:
                utils.log('Updating parameters JSON')
                self.params['params'] = utils.convert_model_parameters_to_list(self.model.parameters())
                with open('params/%s.json' % utils.now(), 'w') as f:
                    json.dump(self.params['params'], f)
                self.changed = False
                self.params_updated = True
        timer = Timer(P.update_json_time, self.convert_params)
        timer.daemon = True
        timer.start()

    def compute_accuracy(self):
        if self.params_updated:
            with self.lock:
                utils.log('Computing model accuracy')
                self.accuracy = self.test()
                self.params_updated = False
        timer = Timer(P.test_model_time, self.compute_accuracy)
        timer.daemon = True
        timer.start()

    def update_model(self):
        while True:
            try:
                parameters = utils.parse_model_parameters_from_list(self.queue.get())
                with self.lock:
                    utils.log('Updating model')
                    utils.merge(self.model.parameters(), parameters, P.merge_ratio)
                    self.changed = True
            except ValueError as v:
                print(v, end='\n' + '*' * 40 + '\n')

    def test(self):
        return utils.test(self.model, self.loader)

    def merge(self, parameters):
        self.queue.put_nowait(parameters)

    def load_params(self, lst):
        utils.merge(self.model.parameters(), utils.parse_model_parameters_from_list(lst), 0)
