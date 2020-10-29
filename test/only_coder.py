#!/usr/bin/env python

import torch

def main(dummy_input):
    # model
    model = torch.load('../checkpoints/only_coder.pth')
    model.eval()
    print(model)

    torch.onnx.export(model,
                      dummy_input,
                      "../checkpoints/conv_tasnet/only_coder.onnx",
                      export_params=True)


if __name__ == '__main__':
    dummy_input = torch.rand(1, 88200)
    main(dummy_input)
