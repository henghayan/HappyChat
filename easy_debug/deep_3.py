#!/usr/bin/env python3
import sys

sys.path.append("../")
import os
import argparse
import torch.nn as nn
import time

import deepspeed
from deepspeed.pipe import PipelineModule

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, num_samples=16, num_features=4096 * 2, num_classes=100):
        super().__init__()
        self.num_samples = num_samples

        self.data = torch.randn(num_samples, 512, num_features)
        self.targets = torch.randn(num_samples, 512, num_features)
        # self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=10,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]
    return layers


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        # self.layers = nn.ModuleList([TimeLinear(4096 * 2, 4096 * 2, i, c) for i in range(16)])
        self.layers = nn.ModuleList([nn.Linear(4096 * 2, 4096 * 2) for i in range(12)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


layer1 = Layer(0).to('cuda:0')
layer2 = Layer(1).to(f'cuda:1')
layers = nn.Sequential(layer1, layer2)


def train_pipe(args, part='parameters'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    # net = AlexNet(num_classes=10)
    net = PipelineModule(layers=layers,
                         loss_fn=torch.nn.MSELoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = MyDataset()
    b = list(net.parameters())
    a = [p for p in net.parameters() if p.requires_grad]

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters()],
        training_data=trainset)

    start = time.time()
    for step in range(args.steps):
        results = engine.train_batch()
        print("res", step, results)
    print("use time", time.time() - start)


if __name__ == '__main__':
    args = get_args()

    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    train_pipe(args)
