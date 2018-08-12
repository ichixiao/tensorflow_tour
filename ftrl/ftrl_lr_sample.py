import logging
import time
import tensorflow as tf
import os
import json
import numpy as np 

# partitioned_variable not supported
# tf.enable_eager_execution()

class LRModel(object):
    def __init__(self, batch_size=5, learning_rate=10, l1_weight=0, l2_weight=0):
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.model_size = 10
        self.l2_shrinkage = 1


    def get_dataset(self, data_paths, mode, epochs=1):
        batch_size = self.batch_size
        
        dataset = tf.data.TextLineDataset(data_paths)
        dataset = dataset.repeat()
        def parse(line):
            fields = tf.string_split([line], ",")
            label = tf.string_to_number(fields.values[0], out_type=tf.float32)
            values = tf.string_to_number(fields.values[1:], out_type=tf.int64)
            values = tf.mod(values, self.model_size)

            return label, tf.SparseTensor(indices=fields.indices[1:], values=values, dense_shape=(1,self.model_size))
        # do parse in lazy
        dataset = dataset.map(parse)
        dataset = dataset.batch(batch_size=self.batch_size)
        return dataset


    def build(self, input_paths, epochs=1, mode='train', variable_partitions=8, config=None):
        variable_partitions = 1
        logging.info('build model: mode = %s partitions = %s', mode, variable_partitions)
        self.global_step = tf.train.get_or_create_global_step()

        dataset = self.get_dataset(input_paths, mode=mode, epochs=epochs).repeat()
        dataset = dataset.prefetch(1)
        self.next_batch = dataset.make_one_shot_iterator().get_next()

        label, features = self.next_batch
        
        # Sparse tensor not supported
        # self.per_sample = tf.split(features, self.batch_size) 
        # self.examples = self.next_batch
        # SparseTensor not supported
        # col_sum = tf.reduce_sum(features, 0)
        # where = tf.not_equal(col_sum, 0)
        # indices = tf.where(where)
        self.non_zero_i = features.values
        self.idx, _ = tf.unique(self.non_zero_i)
        self.sorted_idx = tf.contrib.framework.sort(self.idx)
        self.shape = self.sorted_idx.shape

        partitioner = tf.min_max_variable_partitioner(
                max_partitions=variable_partitions,
                min_slice_size=64 << 20)
        
        with tf.variable_scope(
            'linear',
            partitioner=partitioner):
            self.ps_parameters = tf.get_variable(name="psconstants", shape=(3, self.model_size), initializer=tf.zeros_initializer())
        
        # pull partial varibles from ps_parameters
        self.local_parameter = tf.gather(self.ps_parameters, self.sorted_idx, axis=1)
        # keep updating during training
        w_init = tf.reshape(tf.gather(self.local_parameter, [0]), [-1])
        ni_init = tf.reshape(tf.gather(self.local_parameter, [1]), [-1])
        zi_init = tf.reshape(tf.gather(self.local_parameter, [2]), [-1])

        self.w_init_var = tf.Variable(w_init, trainable=False, validate_shape=False)
        self.n_init_var = tf.Variable(ni_init, trainable=False, validate_shape=False)
        self.z_init_var = tf.Variable(zi_init, trainable=False, validate_shape=False)

        # keep clean to get final deltas
        init_w = tf.gather(self.local_parameter, [0])
        init_n = tf.gather(self.local_parameter, [1])
        init_z = tf.gather(self.local_parameter, [2])

        for i in range(self.batch_size):
            self.line = tf.sparse_slice(features, [i,0,0], [i, 1, self.model_size])
            feas = self.line.values
            re_values = tf.zeros_like(feas) + 1
            zeros = tf.zeros_like(feas)
            re_indices = tf.stack([zeros, feas], 1)
            lens = tf.shape(feas, out_type=tf.int32)[0]
            self.init_feas_idx = tf.zeros_like(feas)
            t = tf.constant(0)
            initial_outputs = tf.TensorArray(dtype=tf.int64, size=lens)
            def cond(t, *args):
                return t < lens

            def body(t, sorted_idx, outputs_):
                cur_fea = tf.gather(feas, t)
                cur_index = tf.where(tf.equal(sorted_idx, cur_fea))
                outputs_ = outputs_.write(t, cur_index)

                return t+1, sorted_idx, outputs_

            t, _, outputs = tf.while_loop(cond, body, [t, self.sorted_idx, initial_outputs])

            outputs = outputs.stack()
            self.feas_indics = tf.reshape(outputs, [-1])
            self.w_ii = tf.gather(self.w_init_var, self.feas_indics)
            n_ii = tf.gather(self.n_init_var, self.feas_indics)
            z_ii = tf.gather(self.z_init_var, self.feas_indics)
            
            lower = tf.map_fn(lambda x: self.l2_weight + (self.l2_shrinkage + tf.sqrt(x))/self.learning_rate, n_ii)
            upper = tf.map_fn(lambda x: tf.cond(tf.abs(x) > self.l1_weight, lambda: tf.sign(x) * self.l1_weight - x, lambda: 0.0), z_ii)
            w_new = upper/lower
            logit = tf.reduce_sum(w_new)
            p = tf.sigmoid(logit)
            grad = tf.gather(label, [i]) - p
            sigmai = (tf.sqrt(n_ii + tf.square(grad)) - tf.sqrt(n_ii)) * self.learning_rate
            z_new = z_ii + grad - sigmai * w_new
            n_new = n_ii + tf.square(grad)

            self.w_updated_var = tf.scatter_update(self.w_init_var, self.feas_indics, w_new)
            self.n_updated_var = tf.scatter_update(self.n_init_var, self.feas_indics, n_new)
            self.z_updated_var = tf.scatter_update(self.z_init_var, self.feas_indics, z_new)

        self.w_delta = self.w_updated_var - init_w
        self.n_delta = self.n_updated_var - init_n
        self.z_delta = self.z_updated_var - init_z

        # self.local_parameter.scatter_update(self.w)

    def train(self, session, time_profile=False):
        n_updated_var, n_init = session.run([self.n_updated_var, self.n_init_var])
        print("updated ops: ")
        print(n_updated_var)
        print("tracking n: ")
        print(n_init)

        # Manul Dump for training test
        variable_export_file = os.path.join('~/tracking/ftrl', f'variables_worker.json')
        with open(variable_export_file, 'w') as f:
            for variable in session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                name = variable.name
                values = session.run(variable)
                f.write(str(values))
                f.write('\n')
        


def main():
    model = LRModel(batch_size=3)
    model.build(input_paths='~/dataset/svm/2/svm_gen_2')

    # server = tf.train.Server.create_local_server()

    step = 0
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        sess.run(tf.global_variables_initializer())
       
        while step < 2:
            model.train(session=sess)
            var_fun=model.get_variables
            print('current step at ', step)
            step += 1



if __name__ == '__main__':
    main()
