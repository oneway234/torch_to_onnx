#!/usr/bin/env python

import numpy as np
import os

import torch

from models.conv_tasnet import ConvTasNet
from params.hparams_tasnet import CreateHparams
import onnxruntime
from onnxruntime.datasets import get_example

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def compare(args):
    # Load torch model
    onnx_path = os.path.join(os.getcwd(), 'checkpoints/conv_tasnet.onnx')
    model = ConvTasNet.load_model(args.model_path)
    model.eval()

    # input data
    dummy_input = torch.rand(1, 88200)

    # Forward
    estimate_source = model(dummy_input)  # [B, C, T]

    # load onnx model
    example_model = get_example(onnx_path)
    session = onnxruntime.InferenceSession(example_model)
    session.get_modelmeta()
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: to_numpy(dummy_input)})

    estimate_source = to_numpy(estimate_source).tolist()
    print("estimate_source", estimate_source)
    print("onnx_out", onnx_out)

    np.testing.assert_almost_equal(estimate_source[0][0], onnx_out[0][0][0], decimal=3)


if __name__ == '__main__':
    hparams = CreateHparams()
    compare(hparams)
