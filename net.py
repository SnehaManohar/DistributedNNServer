from torch import nn

import utils


class NeuralNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = nn.Sequential(*(utils.create_object(**layer) for layer in config['features']['layers']))
        self.reshape = config['reshape']
        self.classifier = nn.Sequential(*(utils.create_object(**layer) for layer in config['classifier']['layers']))
        self.loss = utils.create_object(**config['loss'])

    def forward(self, x):
        return self.classifier(self.features(x).view(*self.reshape))
