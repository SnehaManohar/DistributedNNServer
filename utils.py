from datetime import datetime

import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

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


def log(*args, **kwargs):
    print('[%s]' % (datetime.now().strftime('%Y-%m-%d %I:%M:%S%p')), *args, flush=True, **kwargs)


def empty_log(*_, **__):
    pass


def convert_model_parameters_to_list(parameters):
    return list(map(lambda p: p.data.tolist(), parameters))


def parse_model_parameters_from_list(data):
    return map(np.array, data)


def get_data(training_data=True):
    if P.download:
        CIFAR10(P.data_dir, download=True)
        P.download = False
    return CIFAR10(P.data_dir, transform=P.transform, train=training_data)


def get_train_data():
    return DataLoader(get_data(True), batch_size=P.train_batch_size, shuffle=True)


def get_test_data():
    return DataLoader(get_data(False), batch_size=P.test_batch_size, shuffle=True)


def train(model, loader, optimizer, log_):
    model.train()
    log_('Training...')
    total_loss = 0
    for batch, (data, target) in enumerate(loader, 1):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, target)
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if (batch % P.log_interval) == 0:
            log_('[%d] Average loss: %.4f' % (batch, total_loss / batch))
    log_('Finished training.')
    log_('Average loss: %.4f' % (total_loss / len(loader)))


def test(model, loader, log_):
    test_time = datetime.now()
    model.eval()
    log_('Testing...')
    correct = total = 0
    for batch, (data, target) in enumerate(loader, 1):
        data, target = Variable(data), Variable(target)
        output = model(data)
        correct += (output.max(1)[1] == target).sum().data[0]
        total += len(target)
        if (batch % P.log_interval) == 0:
            log_('[%d] Accuracy: %.2f%% (%d/%d)' % (batch, correct * 1e2 / total, correct, total))
    log_('Finished testing.')
    log_('Accuracy: %.2f%% (%d/%d)' % (correct * 1e2 / total, correct, total))
    return test_time, correct, total
