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
        with open('config.json') as config_file:
            self.config = json.load(config_file)
        self.model = NeuralNet(self.config)
        self.config['optimizer']['parameters']['params'] = self.model.parameters()
        self.optimizer = utils.create_object(**self.config['optimizer'])
        self.params = None
        self.lock = Lock()
        self.changed = True

        self.train_loader = utils.get_train_data()
        self.test_loader = utils.get_test_data()

        self.pre_train(P.epochs)
        self.convert_params()
        Thread(target=self.update_model, daemon=True).start()

    def get_parameters(self):
        return self.params

    def get_queue(self):
        return self.queue

    def pre_train(self, epochs):
        for e in range(1, epochs + 1):
            utils.log('Epoch (%d/%d)' % (e, epochs))
            self.train(verbose=P.verbose_pre_training)
            self.test(verbose=P.verbose_pre_training)
        utils.log('Finished training')

    def convert_params(self):
        if self.changed:
            with self.lock:
                utils.log('Updating parameters JSON')
                self.params = utils.convert_model_parameters_to_list(self.model.parameters())
                self.changed = False
        timer = Timer(P.update_json_time, self.convert_params)
        timer.daemon = True
        timer.start()

    def update_model(self):
        while True:
            parameters = utils.parse_model_parameters_from_list(self.queue.get())
            with self.lock:
                utils.log('Updating model')
                for old, new in zip(self.model.parameters(), parameters):
                    old.data.mul_(1 - P.merge_ratio).add_(torch.FloatTensor(new * P.merge_ratio))
                self.changed = True

    def test(self, verbose=False):
        return utils.test(self.model, self.test_loader, (utils.log if verbose else utils.empty_log))

    def train(self, verbose=False):
        utils.train(self.model, self.train_loader, self.optimizer, (utils.log if verbose else utils.empty_log))
