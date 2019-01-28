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
from utils import encoder, decoder,RNNdecoder,RNNdecoder_inference
from generator import Generator
from copy import deepcopy
from progressbar import ProgressBar
import os
import random
from sklearn.metrics import r2_score
import time
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
    south_list_x[south_list_x>=x_now] = 0
    north_list_x[north_list_x<=x_now] += north_list_x.max()
    west_list_y[west_list_y <= y_now] += west_list_y.max()
    east_list_y[east_list_y >= y_now] = 0
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

    rnn_total = []

    i=0

    def foo(t):
        x = np.ones((t, 69))
        x = np.pad(x, pad_width=((0, 8 - t), (0, 0)), mode='constant')
        return x # 8,69

    while i<= (input_.shape[0]-timestep):
        data=(gen_loc_map(input_,i))
        base_label=np.array([x for x in label_[i:(i+timestep),:]]) # 8,69
        for t in range(2,timestep):
            mask = foo(t)
            for idx in range(0,timestep-t):
                label = base_label[idx:idx+t] # t,69
                label = np.pad(label, pad_width=((0,8-t),(0,0)), mode='constant')
                rnn_total.append((data,label,mask))

            # k=j+1
            # while k<=timestep:
            #
            #     mask = deepcopy(label)
            #     mask[:, :] = 0
            #     mask[j:k,:] = 1
            #     rnn_total.append((data, label, mask))
            #     k+=1

        # rnn_total.append((data,label))
        i+=timestep
        #n,((3,3,2),(t,69))

    return rnn_total

def shuffle_and_transform(rnn_total,mode,num_data=10000,validation_data=-400,test_data=-100):
    if mode == "train":
        rnn_label = []
        rnn_data = []
        rnn_mask = []
        #TODO rewrite input way
        rnn_train = rnn_total[:num_data]

        random.shuffle(rnn_train)
       # (n, 3, 3, 2) + (n, t, 69)
        for item in rnn_train:
            rnn_data.append(item[0])
            rnn_label.append(item[1])
            rnn_mask.append(item[2])
        inputs, labels, masks = np.array(rnn_data, dtype=np.float32), np.array(rnn_label, dtype=np.float32),\
                                np.array(rnn_mask, dtype=np.float32)
        max_label = labels.max(axis=0)
        min_label = labels.min(axis=0)
        mean_label = labels.mean(axis=0)
        labels = (labels - mean_label) / (max_label - min_label+1e-4)
        inputs_and_labels_train = inputs,labels,masks
        return inputs_and_labels_train,(max_label,min_label,mean_label)
    elif mode =="val":
        rnn_label = []
        rnn_data = []
        rnn_mask = []
        rnn_val = rnn_total[validation_data:test_data]
        for item in rnn_val:
            rnn_data.append(item[0])
            rnn_label.append(item[1])
            rnn_mask.append(item[2])
        inputs, labels, masks = np.array(rnn_data, dtype=np.float32), np.array(rnn_label, dtype=np.float32),\
                                np.array(rnn_mask, dtype=np.float32)
        max_label = labels.max()
        min_label = labels.min()
        mean_label = labels.mean()
        labels = (labels - mean_label) / (max_label - min_label)
        inputs_and_labels_val = inputs, labels ,masks
        return inputs_and_labels_val, (max_label, min_label, mean_label)
    elif mode =="test":
        rnn_label = []
        rnn_data = []
        rnn_mask = []
        rnn_val = rnn_total[test_data:]
        for item in rnn_val:
            rnn_data.append(item[0])
            rnn_label.append(item[1])
            rnn_mask.append(item[2])
        inputs, labels, masks = np.array(rnn_data, dtype=np.float32), np.array(rnn_label, dtype=np.float32), \
                                np.array(rnn_mask, dtype=np.float32)
        max_label = labels.max()
        min_label = labels.min()
        mean_label = labels.mean()
        labels = (labels - mean_label) / (max_label - min_label)
        inputs_and_labels_val = inputs, labels, masks
        return inputs_and_labels_val, (max_label, min_label, mean_label)


class VAE(Generator):



    def __init__(self, HIDDEN_SIZE, LEN_OF_YEAR,NUM_OF_CLASSES,HWC_INPUT,learning_rate=0.001,batch_size=1,MODE = "test"):
        self.hwc_input = HWC_INPUT
        self.years = LEN_OF_YEAR
        self.n_classes = NUM_OF_CLASSES
        self.num_rnn_units = 16
        self.restore_model = False
        self.hidden_size = HIDDEN_SIZE
        self.mode = MODE #or train
        if self.mode == "train":
            self.input_tensor = (tf.placeholder(tf.float32, [batch_size, self.hwc_input[0]*self.hwc_input[1],self.hwc_input[2]]),
                                tf.placeholder(tf.float32, [batch_size, self.years, self.n_classes]),
                                tf.placeholder(tf.float32, [batch_size, self.years, self.n_classes]),
                                tf.placeholder(tf.float32,[batch_size,self.years,self.n_classes]))
        elif self.mode == "test":
            self.input_tensor = (
            tf.placeholder(tf.float32, [batch_size, self.hwc_input[0] * self.hwc_input[1], self.hwc_input[2]]),
            tf.placeholder(tf.float32, [batch_size, self.years, self.n_classes]),
            tf.placeholder(tf.float32, [batch_size, self.years, self.n_classes]),
            tf.placeholder(tf.float32, [None]))

        # self.label_tensor = tf.placeholder(tf.float32, [None, 8, 69])
        self.lr = learning_rate


        output_tensor, label_data, mean,stddev = self.model_establish()

        self.vae = vae_loss = self.__get_vae_cost(mean, stddev)  # one to minimize the kl divergence
        rec_loss = self.__get_reconstruction_cost(
            output_tensor, label_data, self.mask)  # the other is to maximize the log likelihood
        v = tf.reduce_sum(tf.square(label_data-tf.reduce_mean(label_data)))
        self.r_square = 1 - rec_loss/v


        # TODO
        loss = vae_loss + rec_loss
        self.loss = loss

        tf.summary.scalar('loss', loss)

        self.merged = tf.summary.merge_all()

                # [batchsize,num_of_years,num_of_drugs]

                # output_tensor = decoder(input_sample)

            # with tf.variable_scope("model", reuse=True) as scope:
            #     self.sampled_tensor = RNNdecoder(tf.random_normal(
            #         [batch_size, hidden_size]))
        self.train = layers.optimize_loss(loss, global_step=self.global_step, learning_rate=tf.constant(self.lr), optimizer='Adam')#, update_ops=[])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess=tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())

    def model_establish(self):
        with arg_scope([layers.conv2d, layers.conv2d_transpose],
                       activation_fn=tf.nn.elu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params={'scale': True}):
            with tf.variable_scope("model") as scope:
                global_step = tf.Variable(0, trainable=False)
                self.global_step = global_step
                #input_[b, h*w ,c],label [b,t,n_classes]
                input_data,label_data,label_mask,label_data_last = self.input_tensor
                self.mask = label_mask
                tf.summary.histogram("target",label_data)
                # label_data = self.label_tensor
                # why is hiddensize timed by 2, get it: outputing mu & sigma, each length hidden_size
                encoded = encoder(input_data, self.hidden_size * 2)
                tf.summary.histogram("encoded",encoded)
                #encoded is a tensor
                mean = encoded[:, :self.hidden_size] # mu
                stddev = tf.sqrt(tf.exp(encoded[:, self.hidden_size:])) # sigma

                epsilon = tf.random_normal([tf.shape(mean)[0], self.hidden_size])
                input_sample = mean + epsilon * stddev # shape? b,h
                # TODO add location to do feature confusion with input_sample
                #  input (b,64)
                tf.summary.histogram("z",input_sample)
                # TODO: explicityly pass in the params here
                # make fusion of z and time

                # loss = tf.multiply(label_data_last,label_mask)
                output_tensor = RNNdecoder_inference(input_sample,label_data_last,N_CLASSES=self.n_classes,
                                           NUM_UNITS=self.num_rnn_units,len_of_year=self.years,mode=self.mode) # TODO why is the reshape necessary
                self.output = output_tensor


        return output_tensor, label_data, mean, stddev

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

    def __get_reconstruction_cost(self, output_tensor, target_tensor, mask, epsilon=1e-8):
        '''Reconstruction loss

        Cross entropy reconstruction loss

        Args:
            output_tensor: tensor produces by decoder
            target_tensor: the target tensor that we want to reconstruct
            epsilon:
        '''
        # TODO: why not mse
        # tf.squared_difference or tf.losses.mean_squared_error
        # tf.reduce_mean
        return tf.reduce_mean(tf.multiply(mask, tf.squared_difference(output_tensor,target_tensor)))

    def update_params(self, input_tensor):
        '''Update parameters of the network

        Args:
            input_tensor: a batch of flattened images [batch_size, 28*28]

        Returns:
            Current loss value
        '''
        return self.sess.run([self.train,self.merged, self.global_step, self.vae], {self.input_tensor: input_tensor})

    def train_model(self,FLAGS,datapath):
        saver = tf.train.Saver(max_to_keep=5)
        if self.restore_model == True:
            # saver =  tf.train.import_meta_graph(self.model_path)
            saver.restore(self.sess, "./save/my-model-23511")

        inputs_and_labels_origin = data_preprocess(load_data(datapath))
        log_path = "./log_{}".format(time.strftime("%d%H%M%S"))
        print(log_path)
        os.makedirs(log_path,exist_ok=True)
        train_writer = tf.summary.FileWriter(log_path)

        train_writer.add_graph(tf.get_default_graph())

        for epoch in range(FLAGS.max_epoch):
            #shuffle
            # already shuffled in data_preprocess
            inputs_and_labels, _ = shuffle_and_transform(inputs_and_labels_origin,"train")
            training_loss = 0.0
             # (n_samples, 3,3,2),(n_samples,t,n_classes)
            num_of_data = inputs_and_labels[0].shape[0]  # num of location maps

            # num_of_data/batch_size
            num_of_batch = math.floor(num_of_data/FLAGS.batch_size)
            # print(num_of_batch)
            for i in range(num_of_data):
                if (i + 1) * FLAGS.batch_size < num_of_data:
                    # now the labels are arbitary time-length
                    inputs, labels, masks = inputs_and_labels
                    label_last_time = deepcopy(labels)
                    # create its last-year label
                    label_last_time[:,1:,:] = label_last_time[:,:-1,:]
                    label_last_time[:,0,:] = 0.0
                    input_and_label = (np.reshape(inputs[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :, :],
                                                  (FLAGS.batch_size, self.hwc_input[0] * self.hwc_input[1], -1)),
                                       labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :],
                                       masks[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :],
                                       label_last_time[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :])
                # input_and_label = input,label
                # input = (batch_size, loc_map)
                # loc_map: tensor (3,3,features)
                # feature = n-vector
                # label = ( batch_size, time_step(8), num_of_drugs(69) )

                    ret = self.update_params(input_and_label)
                    loss_value, summary, step, vaeloss = ret

                    if step% 300 ==0:
                        train_writer.add_summary(summary, global_step=step)
                        print(f'{step}: {loss_value}, {vaeloss}')
                    training_loss += loss_value
                else:
                    break
            training_loss = training_loss / (num_of_batch * FLAGS.batch_size)
            # TODO validation + extra
            print("Epoch{} : Loss {}".format(epoch,training_loss))
            if epoch % 5 ==0:
                saver.save(self.sess,"./save/my-model",global_step=self.global_step)
            self.validation_model(datapath,FLAGS)


        saver.save(self.sess,"./save/my-model-final",global_step=self.global_step)

    def validation_model(self,datapath,FLAGS):
        # saver = tf.train.Saver()
        # saver.restore(self.sess, model_path)

        inputs_and_labels_origin = data_preprocess(load_data(datapath))
        validation_loss = 0.0
        validation_r_square = 0.0
        #size_msg = max,min,mean , to get real output
        inputs_and_labels, size_msg = shuffle_and_transform(inputs_and_labels_origin,mode="val")
        # (n_samples, 3,3,2),(n_samples,t,n_classes)
        num_of_data = inputs_and_labels[0].shape[0]  # num of location maps
        num_of_batch = math.floor(num_of_data / FLAGS.batch_size)
        # tf.reset_default_graph()# ?
        print(num_of_batch)
        for i in range(num_of_data):
            if (i + 1) * FLAGS.batch_size < num_of_data:
                # now the labels are arbitary time-length
                inputs, labels, masks = inputs_and_labels
                label_last_time = deepcopy(labels)
                # create its last-year label
                label_last_time[:, 1:, :] = label_last_time[:, :-1, :]
                label_last_time[:, 0, :] = 0.0
                input_and_label = (np.reshape(inputs[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :, :],
                                              (FLAGS.batch_size, self.hwc_input[0] * self.hwc_input[1], -1)),
                                   labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :],
                                   masks[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :],
                                   label_last_time[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :])

                target = labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :] * masks[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :]

                output,loss_value = self.sess.run([self.output,self.loss],{self.input_tensor:input_and_label})
                # output = output*(size_msg[0]-size_msg[1]+1e-4)+size_msg[2]
                # TODO the output is quite strange
                # print("sample:  ",output[0])

                # r_square = 0.
                # for j in range(FLAGS.batch_size):
                #     r_square +=
                # print(np.mean([r2_score(target[:,t],output[:,t]) for t in range(8)]), np.mean(np.abs(target-output)))
                # print(r_square)
                # validation_r_square += r_square
                validation_loss += loss_value
        validation_loss/=(num_of_batch * FLAGS.batch_size)
        # validation_r_square /= (num_of_batch * FLAGS.batch_size)
        print("validation_loss: ",validation_loss)
        # print("validation_r_square: ",validation_r_square)
        print("target-output: ",np.mean(target-output))

    def inference(self,datapath,FLAGS,model_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

        inputs_and_labels_origin = data_preprocess(load_data(datapath))


        # shuffle
        # already shuffled in data_preprocess
        inputs_and_labels, size_msg = shuffle_and_transform(inputs_and_labels_origin, "test")
        test_loss = 0.0
        # (n_samples, 3,3,2),(n_samples,t,n_classes)
        num_of_data = inputs_and_labels[0].shape[0]  # num of location maps

        # num_of_data/batch_size
        num_of_batch = math.floor(num_of_data / FLAGS.batch_size)
        for i in range(num_of_data):
            if (i + 1) * FLAGS.batch_size < num_of_data:
                # now the labels are arbitary time-length
                inputs, labels, masks = inputs_and_labels
                input_and_label = (np.reshape(inputs[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :, :],
                                              (FLAGS.batch_size, self.hwc_input[0] * self.hwc_input[1], -1)),
                                   labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :],
                                   masks[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :],
                                    np.array([1]))
                # input_and_label = input,label
                # input = (batch_size, loc_map)
                # loc_map: tensor (3,3,features)
                # feature = n-vector
                # label = ( batch_size, time_step(8), num_of_drugs(69) )
                target = labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :] * masks[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :, :]
                output, loss_value = self.sess.run([self.output, self.loss], {self.input_tensor: input_and_label})
                # output = output*(size_msg[0]-size_msg[1]+1e-4)+size_msg[2]
                # print("sample:  ",output[0])
                test_loss += loss_value
            else:
                break
        test_loss /= (num_of_batch * FLAGS.batch_size)
        # validation_r_square /= (num_of_batch * FLAGS.batch_size)
        print("test_loss: ", test_loss)
        # print("validation_r_square: ",validation_r_square)
        print("target-output: ", np.mean(target - output))


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
    flags.DEFINE_integer("max_epoch", 50, "max epoch")
    flags.DEFINE_float("learning_rate", 0.01, "learning rate")
    flags.DEFINE_integer("hidden_size", 64, "size of the hidden VAE unit")
    flags.DEFINE_integer("drug_num",69,"num_of_drug_classes")

    FLAGS = flags.FLAGS

    # if __name__ == "__main__":
    data_path = "../test2.csv"
    hwc_input = (3, 3, 2)
    len_of_years = 8
    # opt = "train"
    opt = "test"
    # model_path= "./save/my-model-14520"
    model_path = "./save/my-model-2420"
    # mnist = input_data.read_data_sets(data_directory, one_hot=True)
    # params: hidden_size, LEN_OF_YEAR,NUM_OF_CLASSES,HWC_INPUT,learning_rate,batch_size

    if opt == "train":
        model = VAE(FLAGS.hidden_size, len_of_years, FLAGS.drug_num, hwc_input, FLAGS.learning_rate, FLAGS.batch_size,MODE=opt)
        model.train_model(FLAGS,data_path)
    else:
        model = VAE(FLAGS.hidden_size, len_of_years, FLAGS.drug_num, hwc_input, FLAGS.learning_rate, FLAGS.batch_size,MODE=opt)
        model.inference(data_path,FLAGS,model_path)

    # origin_data = load_data("../test2.csv")
    # input_data,input_label = data_preprocess(origin_data)
    # print(input_data[0])



            # model.generate_and_save_images(
            #     FLAGS.batch_size, FLAGS.working_directory)

if __name__ == '__main__':
    main()