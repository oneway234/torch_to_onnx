#  -*- coding: utf-8 -*-
import tensorflow as tf


def CreateHparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    # 默认参数
    hparams = tf.contrib.training.HParams(
        ################################
        # Preprocess Parameters        #
        ################################
        in_dir='../CONV_TASNET/example_samples/data',
        out_dir_pre='../CONV_TASNET/example_samples/data',

        ################################
        # Train Parameters        #
        ################################
        train_dir='../CONV_TASNET/example_samples/conv_tasnet/data',
        valid_dir='../CONV_TASNET/example_samples/conv_tasnet/data',
        sample_rate=44100,
        segment=2,  # seconds
        cv_maxlen=6,  # seconds
        N=256,
        L=20,
        B=256,
        H=512,
        P=3,
        X=8,
        R=4,
        C=2,
        norm_type='gLN',
        causal=0,
        mask_nonlinear='relu',
        epochs=10000,
        half_lr=1,
        early_stop=1,
        max_norm=5,
        shuffle=1,
        batch_size=1,
        num_workers=10,
        optimizer='adam',
        lr=1e-4,
        momentum=0,
        l2=0,
        save_folder='../CONV_TASNET/checkpoints/conv_tasnet',
        checkpoint=0,
        continue_from='',
        print_freq=10,
        visdom=0,
        visdom_epoch=0,
        visdom_id="Conv-TasNet Training",

        ################################
        # Separate Parameters             #
        ################################
        model_path='../CONV_TASNET/checkpoints/final.pth.tar',
        mix_dir='../CONV_TASNET/example_samples',
        mix_json='',
        out_dir='../CONV_TASNET/example_samples/demo',

        ################################
        # Common Parameters             #
        ################################
        use_cuda=0,

    )

    # 解析参数列表并更新默认参数
    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
