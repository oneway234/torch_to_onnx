#!/usr/bin/env python
# Created on 2018/12

import argparse
import json
import os
import third_party.librosa as librosa

from params.hparams_tasnet import CreateHparams


def preprocess_dir(in_dir, out_dir, sample_rate=8000):
    file_infos = []
    file_infos_s1 = []
    file_infos_s2 = []
    in_dir = os.path.abspath(in_dir)
    in_dir = os.path.join(in_dir, 'mix')
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.mp3'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        wav_path_s1 = wav_path.replace('mix', 's1')
        wav_path_s2 = wav_path.replace('mix', 's2')
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        samples_s1, _ = librosa.load(wav_path_s1, sr=sample_rate)
        samples_s2, _ = librosa.load(wav_path_s2, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
        file_infos_s1.append((wav_path_s1, len(samples_s1)))
        file_infos_s2.append((wav_path_s2, len(samples_s2)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, 'mix.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)
    with open(os.path.join(out_dir, 's1.json'), 'w') as f:
        json.dump(file_infos_s1, f, indent=4)
    with open(os.path.join(out_dir, 's2.json'), 'w') as f:
        json.dump(file_infos_s2, f, indent=4)


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.mp3'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)


# def preprocess(args):
#     for data_type in ['tr', 'cv', 'tt']:
#         for speaker in ['mix', 's1', 's2']:
#             preprocess_one_dir(os.path.join(args.in_dir, data_type, speaker),
#                                os.path.join(args.out_dir, data_type),
#                                speaker,
#                                sample_rate=args.sample_rate)

def preprocess(args):
    for data_type in ['tr', 'cv', 'tt']:
        preprocess_dir(os.path.join(args.in_dir, data_type),
                           os.path.join(args.out_dir_pre, data_type),
                           sample_rate=args.sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSJ0 data preprocessing")
    parser.add_argument('--in-dir', type=str, default=None,
                        help='Directory path of wsj0 including tr, cv and tt')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Directory path to put output files')
    parser.add_argument('--sample-rate', type=int, default=8000,
                        help='Sample rate of audio file')
    args = parser.parse_args()
    print(args)
    hparams = CreateHparams()
    preprocess(hparams)
