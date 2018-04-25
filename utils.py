from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from parameters import Parameters as P

mapping = {
    # neural network layers
    'Convolution': nn.Conv2d,
    'Linear': nn.Linear,
    'MaxPool': nn.MaxPool2d,
    # activation functions
    'LogSoftMax': nn.LogSoftmax,
    'ReLU': nn.ReLU,
    # loss functions
    'NLLLoss': nn.NLLLoss,
    # optimizer
    'Adam': optim.Adam,
    'SGD': optim.SGD
}


def create_object(name, parameters):
    return mapping[name](**parameters)


def now():
    return datetime.now().strftime('%Y-%m-%d %I:%M:%S%p')


def log(*args, **kwargs):
    print('[%s]' % (now()), *args, flush=True, **kwargs)


def convert_model_parameters_to_list(parameters):
    return list(map(lambda p: p.data.numpy().flatten().tolist(), parameters))


def parse_model_parameters_from_list(data):
    return map(lambda x: np.array(x).astype('float'), data)


def get_data():
    return DataLoader(ImageFolder(P.data_dir, transform=P.transform), batch_size=P.batch_size)


def test(model, loader):
    model.eval()
    log('Testing...')
    correct = total = 0
    for batch, (data, target) in enumerate(loader, 1):
        data, target = Variable(data), Variable(target)
        output = model(data)
        correct += (output.max(1)[1] == target).sum().data[0]
        total += len(target)
        if (batch % P.log_interval) == 0:
            log('[%d] Accuracy: %.2f%% (%d/%d)' % (batch, correct * 1e2 / total, correct, total))
    log('Finished testing.')
    log('Accuracy: %.2f%% (%d/%d)' % (correct * 1e2 / total, correct, total))
    return correct, total


def merge(old, new, alpha):
    for old, new in zip(old, new):
        old.data.mul_(alpha).add_(torch.FloatTensor(new * (1 - alpha)).resize_as_(old.data))
