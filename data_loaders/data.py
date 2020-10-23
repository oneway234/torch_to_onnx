"""
Logic:
1. AudioDataLoader从AudioDataset中产生一个minibatc。AudioDataLoader的batchsize
   是1。真实的batchsize是AudioDataset中的__init__(...)

2. AudioDataLoader从AudioDataset获取一个minibatch后，通过collate_fn(batch)处理数据


Input:
    Mixtured wav tr, cv and tt path
Output:
    One batch at a time.
    Each inputs's shape is B x T
    Each targets's shape is B x C x T
"""

import json
import math
import os

import numpy as np
import torch
import torch.utils.data as data

import third_party.librosa as librosa


class AudioDataset(data.Dataset):

    def __init__(self, json_dir, batch_size, sample_rate=8000, segment=4.0, cv_maxlen=8.0):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioDataset, self).__init__()
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)

        # sort it by #samples (impl bucket)
        def sort(infos):
            return sorted(
                infos, key=lambda info: int(info[0]), reverse=True)

        sorted_mix_infos = mix_infos
        sorted_s1_infos = s1_infos
        sorted_s2_infos = s2_infos

        #从每个json文件中读取指定片段长度wav，wav长度随机
        if segment >= 0.0:
            # segment length and count dropped utts
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples
            drop_utt, drop_len = 0, 0
            for _, sample in sorted_mix_infos:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample
            print("Drop {} utts({:.2f} h) which is short than {} samples".format(
                drop_utt, drop_len / sample_rate / 36000, segment_len))
            # generate minibach infomations
            minibatch = []
            end = 0
            anchor = 0
            while end < len(sorted_mix_infos):
                while anchor <= sorted_mix_infos[end][1]:
                    start = anchor
                    utt_len = int(sorted_mix_infos[end][1] - start)
                    # num_segments = math.floor(utt_len / segment_len)
                    utt_s1 = int(sorted_s1_infos[end][1] - start)
                    utt_s2 = int(sorted_s2_infos[end][1] - start)
                    num_segments = min(math.floor(utt_len / segment_len),
                                       math.floor(utt_s1 / segment_len),
                                       math.floor(utt_s2 / segment_len))
                    if num_segments >= batch_size:
                        anchor += batch_size * segment_len
                        minibatch.append([sorted_mix_infos[end], sorted_s1_infos[end],
                                          sorted_s2_infos[end], sample_rate, segment_len, start, anchor])
                    else:
                        break
                anchor = 0
                end += 1
            self.minibatch = minibatch


        else:  # Load full utterance but not segment
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                end = min(len(sorted_mix_infos), start + batch_size)
                # Skip long audio to avoid out-of-memory issue
                if int(sorted_mix_infos[start][1]) > cv_maxlen * sample_rate:
                    start = end
                    continue
                minibatch.append([sorted_mix_infos[start:end],
                                  sorted_s1_infos[start:end],
                                  sorted_s2_infos[start:end],
                                  sample_rate, segment])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, sources = load_mixtures_and_sources(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    return mixtures_pad, ilens, sources_pad


# Eval data part
from signal_processors.preprocess import preprocess_one_dir


class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        # assert mix_dir != None or mix_json != None
        if mix_dir is not None:
            # Generate mix.json given mix_dir
            preprocess_one_dir(mix_dir, mix_dir, 'mix',
                               sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)

        # sort it by #samples (impl bucket)
        def sort(infos):
            return sorted(
                infos, key=lambda info: int(info[1]), reverse=True)

        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        segment = 2.0
        segment_len = int(segment * sample_rate)
        start = 0
        end = 0
        anchor = 0
        while end < len(sorted_mix_infos):
            while anchor <= sorted_mix_infos[end][1]:
                start = anchor
                utt_len = int(sorted_mix_infos[end][1] - start)
                # num_segments = math.floor(utt_len / segment_len)
                num_segments = math.floor(utt_len / segment_len)
                if num_segments >= batch_size:
                    anchor += batch_size * segment_len
                    minibatch.append([sorted_mix_infos[end], sample_rate, segment_len, start, anchor])
                else:
                    break
            anchor = 0
            end += 1
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])
    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, filenames


# ------------------------------ utils ------------------------------------
def load_mixtures_and_sources(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources = [], []
    # mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch
    mix_infos, s1_infos, s2_infos, sample_rate, segment_len, start, end = batch
    mix_path = mix_infos[0]
    s1_path = s1_infos[0]
    s2_path = s2_infos[0]

    # assert mix_infos[1] == s1_infos[1] and s1_infos[1] == s2_infos[1]
    # read wav file
    try:
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mix = mix[start: end]
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s1 = s1[start: end]
        s2, _ = librosa.load(s2_path, sr=sample_rate)
        s2 = s2[start: end]
    except:
        mix = np.zeros(segment_len)
        s1 = np.zeros(segment_len)
        s2 = np.zeros(segment_len)
    # merge s1 and s2
    s = np.dstack((s1, s2))[0]  # T x C, C = 2
    utt_len = mix.shape[-1]
    if segment_len >= 0:
        # segment
        for i in range(0, utt_len - segment_len + 1, segment_len):
            mixtures.append(mix[i:i + segment_len])
            sources.append(s[i:i + segment_len])
        if utt_len % segment_len != 0:
            mixtures.append(mix[-segment_len:])
            sources.append(s[-segment_len:])
    else:  # full utterance
        mixtures.append(mix)
        sources.append(s)
    return mixtures, sources


def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate, segment_len, start, end = batch
    mix_path = mix_infos[0]
    mix, _ = librosa.load(mix_path, sr=sample_rate)
    mix = mix[start: end]
    utt_len = mix.shape[-1]
    if segment_len >= 0:
        # segment
        for i in range(0, utt_len - segment_len + 1, segment_len):
            mixtures.append(mix[i:i + segment_len])
            filenames.append(mix_path)
        if utt_len % segment_len != 0:
            mixtures.append(mix[-segment_len:])
    else:  # full utterance
        mixtures.append(mix)

    return mixtures, filenames


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    import sys

    json_dir, batch_size = sys.argv[1:3]
    dataset = AudioDataset(json_dir, int(batch_size))
    data_loader = AudioDataLoader(dataset, batch_size=1,
                                  num_workers=4)
    for i, batch in enumerate(data_loader):
        mixtures, lens, sources = batch
        print(i)
        print(mixtures.size())
        print(sources.size())
        print(lens)
        if i < 10:
            print(mixtures)
            print(sources)
