from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework import get_global_step
from utils import encoder, decoder,RNNdecoder
from generator import Generator
from copy import deepcopy
from progressbar import ProgressBar
import os


#10-17, time step = 8

def gen_loc_map(loc_data,idx):
    #loc_data = (n,2)
    #return (3,3,2)
    loc_map = np.zeros((3,3,2))
    i = loc_data[idx]
    # each row -> map
    x_now, y_now = i[0],i[1]
    north_list_x, south_list_x = deepcopy(loc_data[:,0]),deepcopy(loc_data[:,0])
    west_list_y, east_list_y = deepcopy(loc_data[:,1]),deepcopy(loc_data[:,1])
    south_list_x[south_list_x>x_now] = 0
    north_list_x[north_list_x<x_now] += north_list_x.max()
    west_list_y[west_list_y < y_now] += west_list_y.max()
    east_list_y[east_list_y > y_now] = 0
    south = np.where(south_list_x==south_list_x.max())[0][0]
    north = np.where(north_list_x==north_list_x.min())[0][0]
    west =np.where(west_list_y==west_list_y.min())[0][0]
    east = np.where(east_list_y==east_list_y.max())[0][0]
    loc_map[1,1] = i
    loc_map[0,1] = loc_data[north]
    loc_map[1,0] = loc_data[west]
    loc_map[1,2] = loc_data[east]
    loc_map[2,1] = loc_data[south]

    return loc_map

# def optimize(optimizer,predict,target):
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=predict)
#     avg_loss = tf.reduce_mean(loss)
#     opt = optimizer.minimize(avg_loss)
#     return avg_loss,opt

def load_data(file_path):
    # 假设file 是 坐标x, 坐标y, 药物矩阵 的形式保存( 8 years/a package
    df = pd.read_csv(file_path)
    # dataset =list()
    # for indexs in df.index:
    #     dataset.append(df.iloc[indexs,:].as_matrix())
    dataset = df.as_matrix()

    # dataset= dataset[1:,:]

    data_input = dataset[:,-2:]
    data_label = dataset[:,6:-5]
    return (data_input,data_label)
    # train_data = data_input[:19000,:]
    # val_data = data_input[19000:,:]
    # train_label = data_label[:19000,:]
    # val_label = data_label[19000:,:]
    #
    # return train_data,train_label,val_data,val_label



def data_preprocess(data,timestep=8):
    #data : dataframe 2d (x1,x2)
    input_, label_ =data

    rnn_data = []
    rnn_label = []
    i=0
    while i<= (input_.shape[0]-timestep):
        rnn_data.append(gen_loc_map(input_,i))
        rnn_label.append([x for x in label_[i:(i+timestep),:]])
        i+=timestep
        #n,3,3,2  + n,8,69
    return np.array(rnn_data,dtype=np.float32),np.array(rnn_label,dtype=np.float32)

# origin_data = load_data("../test2.csv")
# input_data,input_label = data_preprocess(origin_data)
# print(input_data[0])



class VAE(Generator):

    def __init__(self, hidden_size, learning_rate=1e-2,batch_size=1):
        self.input_tensor = tf.placeholder(

            tf.float32, [None, 3*3,2]),tf.placeholder(tf.float32, [None, 8, 69])
        # self.label_tensor = tf.placeholder(tf.float32, [None, 8, 69])
        self.lr = 0.001

        #
        with arg_scope([layers.conv2d, layers.conv2d_transpose],
                       activation_fn=tf.nn.elu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params={'scale': True}):
            with tf.variable_scope("model") as scope:
                global_step = tf.Variable(0, trainable=False)
                self.global_step = global_step
                #input_[tensor batch_size, num_of_location ,2]
                input_data,label_data = self.input_tensor
                # label_data = self.label_tensor
                encoded = encoder(input_data, hidden_size * 2)
                #encoded is a tensor
                mean = encoded[:, :hidden_size]
                stddev = tf.sqrt(tf.exp(encoded[:, hidden_size:]))

                epsilon = tf.random_normal([tf.shape(mean)[0], hidden_size])
                input_sample = mean + epsilon * stddev
                # TODO add location to do feature confusion with input_sample
                #  input (?,64)


                output_tensor = RNNdecoder(tf.reshape(input_sample,[-1,input_sample.shape[-1]]))
                # [batchsize,num_of_years,num_of_drugs]

                # output_tensor = decoder(input_sample)

            # with tf.variable_scope("model", reuse=True) as scope:
            #     self.sampled_tensor = RNNdecoder(tf.random_normal(
            #         [batch_size, hidden_size]))

        vae_loss = self.__get_vae_cost(mean, stddev)
        rec_loss = self.__get_reconstruction_cost(
            output_tensor, label_data)

        loss = vae_loss + rec_loss
        self.train = layers.optimize_loss(loss, global_step=global_step, learning_rate=tf.constant(self.lr), optimizer='Adam')#, update_ops=[])

        self.sess=tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def __get_vae_cost(self, mean, stddev, epsilon=1e-8):
        '''VAE loss
            See the paper

        Args:
            mean:
            stddev:
            epsilon:
        '''
        return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                    2.0 * tf.log(stddev + epsilon) - 1.0))

    def __get_reconstruction_cost(self, output_tensor, target_tensor, epsilon=1e-8):
        '''Reconstruction loss

        Cross entropy reconstruction loss

        Args:
            output_tensor: tensor produces by decoder
            target_tensor: the target tensor that we want to reconstruct
            epsilon:
        '''
        return tf.reduce_sum(tf.square(target_tensor-output_tensor))

    def update_params(self, input_tensor):
        '''Update parameters of the network

        Args:
            input_tensor: a batch of flattened images [batch_size, 28*28]

        Returns:
            Current loss value
        '''
        return self.sess.run(self.train, {self.input_tensor: input_tensor})

    def train_model(self,FLAGS,datapath):
        saver = tf.train.Saver(max_to_keep=5)
        for epoch in range(FLAGS.max_epoch):
            training_loss = 0.0
            inputs_and_labels = data_preprocess(load_data(datapath))
            num_of_data = inputs_and_labels[0].shape[0]  # num of location maps
            for i in range(num_of_data):
                if (i + 1) * FLAGS.batch_size < inputs_and_labels[0].shape[0]:
                    inputs, labels = inputs_and_labels

                    input_and_label = (np.reshape(inputs[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :, :],
                                                  (FLAGS.batch_size, 3 * 3, -1)),
                                       labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :])
                # input_and_label = input,label
                # input = (batch_size, loc_map)
                # loc_map: tensor (3,3,features)
                # feature = n-vector
                # label = ( batch_size, time_step(8), num_of_drugs(69) )
                loss_value = self.update_params(input_and_label)

                training_loss += loss_value

            training_loss = training_loss / (FLAGS.updates_per_epoch * FLAGS.batch_size)
            # TODO model_save + validation + extra
            print("Epoch{} Loss {}".format(epoch, training_loss))
            if epoch %50 ==0:
                saver.save(self.sess,"./save/my-model",global_step=self.global_step)

        saver.save(self.sess,"./save/my-model-final",global_step=self.global_step)



def read_next(inputs_and_labels,id,batch_size):
    if (id+1)*batch_size<inputs_and_labels[0].shape[0]:
        inputs,labels = inputs_and_labels

        return np.reshape(inputs[id*batch_size:(id+1)*batch_size,:,:,:],(batch_size,3*3,-1)),labels[id*batch_size:(id+1)*batch_size,:,:]
    else:
        print("out ot range")
        return



def main():
    flags = tf.flags

    flags.DEFINE_integer("batch_size", 4, "batch size")
    flags.DEFINE_integer("updates_per_epoch", 10, "number of updates per epoch")
    flags.DEFINE_integer("max_epoch", 200, "max epoch")
    flags.DEFINE_float("learning_rate", 0.01, "learning rate")
    flags.DEFINE_integer("hidden_size", 64, "size of the hidden VAE unit")

    FLAGS = flags.FLAGS

    if __name__ == "__main__":
        data_path = "../test2.csv"

        # mnist = input_data.read_data_sets(data_directory, one_hot=True)

        model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)
        model.train_model(FLAGS,data_path)




            # model.generate_and_save_images(
            #     FLAGS.batch_size, FLAGS.working_directory)

main()