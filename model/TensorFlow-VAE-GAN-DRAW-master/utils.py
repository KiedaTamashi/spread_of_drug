import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn


def encoder(input_tensor, output_size,feature_vector_lens=2):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 3,3,feature_vector] b,h,w,c

    Returns:
        A tensor that expresses the encoder network
    '''
    #TODO the figure is too small
    # TODO: why is this reshape necessary
    net = tf.reshape(input_tensor, [-1, 3, 3, feature_vector_lens])
    # TODO: I suggest writing in loop, e.g.
    # ######################
    # outs = (32,64,128)
    # ks = (3,3,3)
    # ss = (1,1,2)
    # filters = zip(outs,ks,ss) # the above should be sent by function args
    # for n_out,k,s in filters:
    #     net = layers.conv2d(net, num_outputs=n_out, kernel_size=k, stride=s)
    # ########################
    net = layers.conv2d(net, 32, 3, stride=1)
    net = layers.conv2d(net, 64, 3, stride=1)
    # TODO I do not think stride=2 may make a difference, the shape is only 3x3
    net = layers.conv2d(net, 128, 3, stride=2, padding='VALID') # h=w=3, b,1,1,128
    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net) # TODO mark the shape here
    return layers.fully_connected(net, output_size, activation_fn=None) # TODO is there a reason why act is not used


def discriminator(input_tensor):
    '''Create a network that discriminates between images from a dataset and
    generated ones.

    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''

    return encoder(input_tensor, 1)


def decoder(input_tensor):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        A tensor that expresses the decoder network
    '''
    #turn to 3d
    net = tf.expand_dims(input_tensor, 1)
    net = tf.expand_dims(net, 1)
    net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
    net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
    net = layers.conv2d_transpose(net, 32, 5, stride=2)
    net = layers.conv2d_transpose(
        net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
    net = layers.flatten(net)
    return net

def RNNdecoder(input_tensor,N_CLASSES=69,NUM_UNITS=16):
    # input_tensor: a batch of vectors to decode
    # 'outputs' is a tensor of shape [batch_size, max_time, NUM_UNITS]

    #TODO not sure
    net = tf.expand_dims(input_tensor, 1) # b,1,h
    # net = tf.expand_dims(net, 1)
    net = tf.tile(net,[1,8,1]) # b,8,h, TODO: remove hardcoded numbers
    rnn_cell = rnn.BasicLSTMCell(num_units=NUM_UNITS)
    # TODO: is dynamic rnn necessary?
    outputs, final_state = tf.nn.dynamic_rnn(
        cell=rnn_cell,  # 选择传入的cell
        inputs=net,  # 传入的数据
        initial_state=None,  # 初始状态
        dtype=tf.float32,  # 数据类型
        time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)，这里根据image结构选择False
    )
    # TODO: check the distribution

    output = tf.layers.dense(inputs=outputs, units=N_CLASSES)
    tf.summary.histogram('output', output)
    # tf.summary.histogram('target', )
    #output: [batchsize,num_of_years,num_of_drugs]
    return output