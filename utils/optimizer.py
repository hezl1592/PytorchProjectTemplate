import torch.optim as optim
from .scheduler import CosineWithRestarts
import torch


def create_optimizer(params, mode='adam', base_lr=1e-3, t_max=10):
    # t_max = 60
    if mode == 'adam':
        optimizer = optim.Adam(params, base_lr)
    elif mode == 'sgd':
        optimizer = optim.SGD(params, base_lr, momentum=0.9, weight_decay=4e-5)
    else:
        raise NotImplementedError(mode)

    scheduler = CosineWithRestarts(optimizer, t_max)

    return optimizer, scheduler


def create_optimizer_(net, cfg):
    training_params = filter(lambda p: p.requires_grad, net.parameters())
    if cfg.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(training_params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(training_params, lr=cfg.base_lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(cfg.optimizer)

    scheduler = CosineWithRestarts(optimizer, cfg.t_max)

    return optimizer, scheduler
