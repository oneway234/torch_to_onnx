# -*- coding:utf-8 -*-
# echo off
# by 张博闻

from test.separate_tasnet import separate
from params.hparams_tasnet import CreateHparams

if __name__ == '__main__':
    """使用 TasNet 对人声伴奏进行分离"""

    hparams = CreateHparams()
    separate(hparams)
