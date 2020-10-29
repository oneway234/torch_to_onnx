#!/usr/bin/env python

import argparse

import torch

from models.solver import Solver
from models.encoder_decoder import ConvTasNet
from params.hparams_tasnet import CreateHparams

def main(dummy_input):
    # Construct Solver

    # model
    model = torch.load('../checkpoints/only_coder.pth')
    print(model)

    # input data
    torch.onnx.export(model.eval(),
                      dummy_input,
                      "../checkpoints/conv_tasnet/only_coder.onnx",
                      export_params=True)


if __name__ == '__main__':
    dummy_input = torch.rand(1, 88200)
    main(dummy_input)
