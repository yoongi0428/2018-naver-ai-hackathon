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

import argparse
import os
import datetime

import numpy as np
import tensorflow as tf

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML

from util.KinQueryDataset import KinQueryDataset, kin_preprocess
from models.kin_cnn import Model


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        q1 = []
        q2 = []
        for line in raw_data:
            lines = line.split('\t')
            q1.append(lines[0])
            q2.append(lines[1])


        preprocessed_data1 = kin_preprocess(q1, config.strmaxlen)
        preprocessed_data2 = kin_preprocess(q2, config.strmaxlen)

        feed_dict = {x1: preprocessed_data1, x2: preprocessed_data2}
        pred = sess.run(output_prob, feed_dict=feed_dict)
        clipped = np.array(pred > config.threshold, dtype=np.int)
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

def infer_test(raw_data, config):
    q1 = []
    q2 = []
    for line in raw_data:
        lines = line.split('\t')
        q1.append(lines[0])
        q2.append(lines[1])

    # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
    preprocessed_data1 = kin_preprocess(q1, config.strmaxlen)
    preprocessed_data2 = kin_preprocess(q2, config.strmaxlen)

    # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
    feed_dict = {x1: preprocessed_data1, x2: preprocessed_data2}

    pred = sess.run(output_prob, feed_dict=feed_dict)
    clipped = np.array(pred > config.threshold, dtype=np.int)

    return clipped.flatten()

def _batch_loader(iterable, n=1):
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def is_better_result(best, cur):
    return best < cur


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=15)
    args.add_argument('--epochs', type=int, default=300)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--lr', type=float, default=0.006)
    args.add_argument('--strmaxlen', type=int, default=150)
    args.add_argument('--charsize', type=int, default=300)
    args.add_argument('--filter_num', type=int, default=256)
    args.add_argument('--emb', type=int, default=256)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()
    if config.eumjeol:
        config.charsize = 2510

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    MODE = 'train'
    DISPLAY_STEP = 100
    SUBTEST_STEP = 10

    fc_hidden = [1500, 500, 100]
    filter_sizes = [4, 5]

    model = Model(config, fc_hidden, filter_sizes)
    model.fit()

    output_prob = model.output_prob
    train_step = model.train_step
    loss = model.loss
    x1 = model.x1
    x2 = model.x2
    y_ = model.y_

    ####################################################################################

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    print('=' * 15 + "MODEL INFO" + '=' * 15)
    print(config)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        print("Loading KIN Dataset")
        dataset = KinQueryDataset(dataset_path=DATASET_PATH, mode=MODE, max_length=config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len // config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1

        # epoch마다 학습을 수행합니다.
        best_result = -1
        best_epoch = -1
        print('=' * 15 + "TRAINING START" + '=' * 15)
        for epoch in range(1, config.epochs + 1):
            avg_loss = 0.0
            for i, data in enumerate(_batch_loader(dataset, config.batch)):
                data1 = data[0]
                data2 = data[1]
                labels = data[2].flatten()

                feed_dict = {x1: data1, x2: data2, y_: labels}

                _, l = sess.run([train_step, loss], feed_dict=feed_dict)

                if i % DISPLAY_STEP == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('[%s] Batch : (%3d/%3d), LOSS in this minibatch : %.3f' % (
                    time_str, i, one_batch_size, float(l)))
                avg_loss += float(l)
            print('epoch:', epoch, ' train_loss:', avg_loss)

            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss / one_batch_size), step=epoch)

            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)