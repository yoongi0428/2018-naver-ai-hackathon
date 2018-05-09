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
import numpy as np

from util.kor_char_parser import decompose_str_as_one_hot


class KinQueryDataset:
    def __init__(self, dataset_path, mode, max_length=420):
        # 데이터, 레이블 각각의 경로
        file_prefix = mode
        queries_data = os.path.join(dataset_path, mode, file_prefix + '_data')
        labels_data = os.path.join(dataset_path, mode, file_prefix + '_label')

        # 지식인 데이터를 읽고 preprocess까지 진행합니다
        with open(queries_data, 'rt', encoding='utf8') as f:
            q1_2 = f.readlines()

            q1 =[]
            q2 =[]
            for line in q1_2:
                lines = line.split('\t')
                q1.append(lines[0])
                q2.append(lines[1])
                
            self.queries1 = kin_preprocess(q1, max_length)
            self.queries2 = kin_preprocess(q2, max_length)

        with open(labels_data) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])


    def __len__(self):
        return len(self.queries1)

    def __getitem__(self, idx):
        return self.queries1[idx], self.queries2[idx], self.labels[idx]

    def shuffle(self):
        p = np.arange(len(self.queries1))
        np.random.shuffle(p)
        self.queries1 = self.queries1[p, :]
        self.queries2 = self.queries2[p, :]
        self.labels = self.labels[p]

def kin_preprocess(data: list, max_length: int):
    vectorized_data = [decompose_str_as_one_hot(datum, warning=False) for datum in data]

    zero_padding = np.zeros((len(data), max_length), dtype=np.int32)

    for idx, seq in enumerate(vectorized_data):
        length = len(seq)
        if length >= max_length:
            length = max_length
            zero_padding[idx, :length] = np.array(seq)[:length]
        else:
            zero_padding[idx, :length] = np.array(seq)
    return zero_padding