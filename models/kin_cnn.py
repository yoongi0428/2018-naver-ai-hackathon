# -*- coding: utf-8 -*-
import tensorflow as tf

class Model():
    def __init__(self, config, output_layers, filter_sizes):
        self.embedding = config.emb
        self.strmaxlen = config.strmaxlen
        self.character_size = 2510
        self.learning_rate = config.lr
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.input_size = self.embedding * self.strmaxlen
        self.output_size = config.output #1

        self.output_layer1 = output_layers[1]
        self.output_layer2 = output_layers[2]
        self.num_filters = config.filter_num
        self.filter_sizes = filter_sizes


    def fit(self):
        self.x1 = tf.placeholder(tf.int32, [None, self.strmaxlen])
        self.x2 = tf.placeholder(tf.int32, [None, self.strmaxlen])
        self.y_ = tf.placeholder(tf.float32, [None])

        char_embedding = tf.get_variable('char_embedding', [self.character_size, self.embedding])

        embedded1 = tf.nn.embedding_lookup(char_embedding, self.x1)
        embedded2 = tf.nn.embedding_lookup(char_embedding, self.x2)

        self.embedded1_expanded = tf.expand_dims(embedded1, -1)
        self.embedded2_expanded = tf.expand_dims(embedded2, -1)

        pooled_outputs1 = []
        pooled_outputs2 = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv1 = tf.nn.conv2d(
                    self.embedded1_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv1")

                conv2 = tf.nn.conv2d(
                    self.embedded2_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2")
                # Apply nonlinearity
                h1 = tf.nn.relu(tf.nn.bias_add(conv1, b), name="relu1")
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu2")
                # Maxpooling over the outputs
                pooled1 = tf.nn.max_pool(
                    h1,
                    ksize=[1, self.strmaxlen - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool1")
                pooled2 = tf.nn.max_pool(
                    h2,
                    ksize=[1, self.strmaxlen - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool2")
                pooled_outputs1.append(pooled1)
                pooled_outputs2.append(pooled2)


        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool1 = tf.concat(pooled_outputs1, 3)
        self.h_pool_flat1 = tf.reshape(self.h_pool1, [-1, num_filters_total])
        self.h_pool2 = tf.concat(pooled_outputs2, 3)
        self.h_pool_flat2 = tf.reshape(self.h_pool2, [-1, num_filters_total])

        with tf.name_scope("output1"):
            W = self.weight_variable([num_filters_total, self.output_layer1])
            b =  self.bias_variable([self.output_layer1])
            self.output_11 = tf.nn.relu(tf.nn.xw_plus_b(self.h_pool_flat1, W, b, name="scores1"))
            self.output_22 = tf.nn.relu(tf.nn.xw_plus_b(self.h_pool_flat2, W, b, name="scores2"))

        with tf.name_scope("output2"):
            W = self.weight_variable([self.output_layer1, self.output_layer2])
            b =  self.bias_variable([self.output_layer2])
            self.output_111 = tf.nn.relu(tf.nn.xw_plus_b(self.output_11, W, b, name="scores21"))
            self.output_222 = tf.nn.relu(tf.nn.xw_plus_b(self.output_22, W, b, name="scores22"))

        with tf.name_scope("output3"):
            W = self.weight_variable([self.output_layer2, self.output_size])
            b =  self.bias_variable([self.output_size])
            self.output_sigmoid1 = tf.nn.xw_plus_b(self.output_111, W, b, name="scores211")
            self.output_sigmoid2 = tf.nn.xw_plus_b(self.output_222, W, b, name="scores222")

        self.output_prob = tf.exp(-tf.reduce_sum(tf.abs(self.output_sigmoid1 - self.output_sigmoid2), axis=1))

        # loss & optimizer
        self.loss = tf.losses.mean_squared_error(self.y_ , self.output_prob)
        opti = tf.train.AdamOptimizer(self.learning_rate)
        gradients, variables = zip(*opti.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 3.0)
        self.train_step = opti.apply_gradients(zip(gradients, variables), global_step=self.global_step)





    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
