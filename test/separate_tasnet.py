#!/usr/bin/env python

import numpy as np
import os

import third_party.librosa as librosa
import torch

from data_loaders.data import EvalDataLoader, EvalDataset
from models.conv_tasnet import ConvTasNet
from signal_processors.utils import remove_pad

def separate(args):
    if args.mix_dir is None and args.mix_json is None:
        print("Must provide mix_dir or mix_json! When providing mix_dir, "
              "  is ignored.")

    # Load model
    model = ConvTasNet.load_model(args.model_path)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    eval_dataset = EvalDataset(args.mix_dir, args.mix_json,
                               batch_size=args.batch_size,
                               sample_rate=args.sample_rate)
    eval_loader = EvalDataLoader(eval_dataset, batch_size=1)
    os.makedirs(args.out_dir, exist_ok=True)

    def write(inputs, filename, sr=args.sample_rate):
        librosa.output.write_wav(filename, inputs, sr, norm=True)  # norm=True)

    with torch.no_grad():
        mix = []
        s1 = []
        s2 = []
        for (i, data) in enumerate(eval_loader):
            # Get batch data
            mixture, mix_lengths, filenames = data
            if args.use_cuda:
                mixture, mix_lengths = mixture.cuda(), mix_lengths.cuda()

            # Forward
            estimate_source = model(mixture)  # [B, C, T]

            # # save ONNX model
            # torch.onnx.export(model, mixture,
            #                   "../CONV_TASNET/checkpoints/conv_tasnet.onnx",
            #                   export_params=True
            #                   # do_constant_folding=True
            #                   )
            # # Remove padding and flat
            flat_estimate = remove_pad(estimate_source, mix_lengths)
            mixture = remove_pad(mixture, mix_lengths)
            # Write result
            last_filename = os.path.join(args.out_dir,
                                         os.path.basename(filenames[0]).strip('.wav'))
           
            
            for i, filename in enumerate(filenames):
                filename = os.path.join(args.out_dir,
                                        os.path.basename(filename).strip('.wav'))
                if filename == last_filename:
                    mix.append(mixture[i])
                    s1.append(flat_estimate[i][0])
                    s2.append(flat_estimate[i][1])
                else:
                    mix = np.array(mix)
                    s1 = np.array(s1)
                    s2 = np.array(s2)                 
                    write(mix, last_filename[0:-4] + '.wav')
                    write(s1, last_filename[0:-4] + '_s1.wav') 
                    write(s2, last_filename[0:-4] + '_s2.wav') 
                    mix = []
                    s1 = []
                    s2 = []
                last_filename = filename
        mix = np.array(mix).flatten()
        s1 = np.array(s1).flatten()
        s2 = np.array(s2).flatten()
        write(mix, last_filename[0:-4] + '.wav')  
        write(s1, last_filename[0:-4] + '_s1.wav')
        write(s2, last_filename[0:-4] + '_s2.wav')
        print('saved at:' + last_filename + '.wav')

