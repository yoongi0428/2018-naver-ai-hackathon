# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
from util.KinQueryDataset import KinQueryDataset, kin_preprocess
from util.MovieReviewDataset import MovieReviewDataset, mv_preprocess
import numpy as np


class HackathonDataset:

    def __init__(self, dataset_path, which, mode='train', local=False, eumjeol=False, phase=1, max_len=420):
        """

        :param dataset_path:
        :param which_data: 0, 1
        :param mode:
        :param eumjeol:
        :param phase:
        :param max_len:
        """
        self.WHICH = which
        file_prefix = mode
        if self.WHICH == 'movie':
            self.Dataset = MovieReviewDataset(dataset_path, mode, local, eumjeol, phase, max_len)
            self.preprocess = mv_preprocess
        else:
            self.Dataset = KinQueryDataset(dataset_path, mode, local, eumjeol, phase, max_len)
            self.preprocess = kin_preprocess
        self.shuffle()

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.Dataset)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        if self.WHICH == 'movie':
            return self.Dataset.reviews[idx], self.Dataset.ratings[idx]
        else:
            return self.Dataset.queries1[idx], self.Dataset.queries2[idx], self.Dataset.labels[idx]

    def shuffle(self):
        self.Dataset.shuffle()

def preprocess(which, data, max_len, eumjeol=False):
    if which == 'movie':
        return mv_preprocess(data, max_len, eumjeol)
    else:
        return kin_preprocess(data, max_len)