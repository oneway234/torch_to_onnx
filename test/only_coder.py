#!/usr/bin/env python

import torch

def main(dummy_input):
    # Construct Solver

    # model
    model = torch.load('../checkpoints/only_coder.pth')
    model.eval()
    print(model)

    # input data
    torch.onnx.export(model,
                      dummy_input,
                      "../checkpoints/conv_tasnet/only_coder.onnx",
                      export_params=True)


if __name__ == '__main__':
    dummy_input = torch.rand(1, 88200)
    main(dummy_input)
