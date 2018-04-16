import json
from queue import Queue
from threading import Lock, Timer, Thread

import torch

import utils
from net import NeuralNet
from parameters import Parameters as P


class Model:
    def __init__(self):
        self.queue = Queue()
        self.train_request_queue = Queue()
        self.train_progress_queue = Queue()
        with open('config.json') as config_file:
            self.config = json.load(config_file)
        self.model = NeuralNet(self.config)
        self.params = self.config['optimizer']['parameters']
        self.params['params'] = self.model.parameters()
        self.parameter_lengths = list(map(lambda x: x.data.nelement(), self.model.parameters()))
        self.optimizer = utils.create_object(**self.config['optimizer'])
        self.accuracy = None
        self.lock = Lock()
        self.changed = True
        self.params_updated = False

        self.train_loader = utils.get_train_data()
        self.test_loader = utils.get_test_data()

        self.pre_train(P.epochs)
        self.convert_params()
        self.compute_accuracy()
        Thread(target=self.update_model, daemon=True).start()
        Thread(target=self.train_model, daemon=True).start()

    def get_parameters(self):
        return self.config

    def get_queue(self):
        return self.queue

    def get_train_queue(self):
        return self.train_request_queue

    def get_progress_queue(self):
        return self.train_progress_queue

    def pre_train(self, epochs):
        for e in range(1, epochs + 1):
            utils.log('Epoch (%d/%d)' % (e, epochs))
            self.train(verbose=P.verbose_pre_training)
            self.test(verbose=P.verbose_pre_training)
        utils.log('Finished training')

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
                    for old, new in zip(self.model.parameters(), parameters):
                        old.data.mul_(1 - P.merge_ratio).add_(torch.FloatTensor(new * P.merge_ratio).resize_as_(old.data))
                    self.changed = True
            except ValueError as v:
                print(v, end='\n' + '*' * 40 + '\n')

    def train_model(self):
        while True:
            self.train_request_queue.get()
            with self.lock:
                self.train_progress_queue.put_nowait(0)
                self.train(P.verbose_training_request)
            self.train_progress_queue.put(1)

    def test(self, verbose=False):
        return utils.test(self.model, self.test_loader, (utils.log if verbose else utils.empty_log))

    def train(self, verbose=False):
        utils.train(self.model, self.train_loader, self.optimizer, (utils.log if verbose else utils.empty_log))
        self.params_updated = True
