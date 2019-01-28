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
    output_units = (32,64)
    kernels = (3,3)
    strides = (1,1)
    paddings = ('SAME','VALID')
    filters = zip(output_units,kernels,strides,paddings)
    for n_out,k,s,p in filters:
        net = tf.layers.batch_normalization(net)
        net = layers.conv2d(net,n_out,k,stride=s,padding=p)

    # ######################
    # outs = (32,64,128)
    # ks = (3,3,3)
    # ss = (1,1,2)
    # filters = zip(outs,ks,ss) # the above should be sent by function args
    # for n_out,k,s in filters:
    #     net = layers.conv2d(net, num_outputs=n_out, kernel_size=k, stride=s)
    # ########################
    # net = layers.conv2d(net, 32, 3, stride=1)
    # net = layers.conv2d(net, 64, 3, stride=1)
    # # TODO I do not think stride=2 may make a difference, the shape is only 3x3
    # net = layers.conv2d(net, 128, 3, stride=1, padding='VALID') # h=w=3, b,1,1,128
    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net) # b,128
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

def RNNdecoder(input_tensor,output_tensor_last,N_CLASSES=69,NUM_UNITS=16,len_of_year = 8):
    # input_tensor: a batch of vectors to decode
    # output_tensor_last: b,len_of_year,n_classes
    # len_of_year should be the time-length processed now
    # 'outputs' is a tensor of shape [batch_size, max_time, NUM_UNITS]

    #TODO not sure
    net = tf.expand_dims(input_tensor, 1) # b,1,h
    # feature fusion
    output_tensor_last = layers.fully_connected(output_tensor_last, int(input_tensor.shape[-1])) # to b,t,h
    net = tf.tile(net, [1, len_of_year, 1])  # b,t,h
    net = tf.concat([net,output_tensor_last],axis=-1)
    tf.summary.histogram('concat', net)
    # net = tf.expand_dims(net, 1)
    rnn_cell = rnn.GRUCell(num_units=NUM_UNITS)
    # TODO: is dynamic rnn necessary?
    outputs, final_state = tf.nn.dynamic_rnn(
        cell=rnn_cell,  # 选择传入的cell
        inputs=net,  # 传入的数据
        initial_state=None,  # 初始状态
        dtype=tf.float32,  # 数据类型
        time_major=False,  # False: (batch, time_step, input); True: (time step, batch, input)，这里根据image结构选择False
    )
    # TODO: check the distribution

    output = tf.layers.dense(inputs=outputs, units=N_CLASSES, activation=tf.nn.tanh)
    tf.summary.histogram('output', output)
    # tf.summary.histogram('target', )
    #output: [batchsize,num_of_years,num_of_drugs]
    return output

def RNNdecoder_inference(input_tensor,N_CLASSES=69,NUM_UNITS=16,len_of_year = 8):
    # input_tensor: a batch of vectors to decode
    # output_tensor_last: b,len_of_year,n_classes
    # len_of_year should be the time-length processed now
    # 'outputs' is a tensor of shape [batch_size, max_time, NUM_UNITS]
    # net = tf.expand_dims(input_tensor, 1)  # b,1,h
    # feature fusion
    # output_tensor_last = layers.fully_connected(output_tensor_last, int(input_tensor.shape[-1]))  # to b,t,h
    # net = tf.tile(net, [1, len_of_year, 1])  # b,t,h
    # net = input_tensor
    # net = tf.concat([net, output_tensor_last], axis=-1)
    tf.summary.histogram('concat', input_tensor)
    # net = tf.expand_dims(net, 1)
    rnn_cell = rnn.GRUCell
    cells = []
    unit_size = [NUM_UNITS]*len_of_year
    initial_state = rnn.GRUCell.zero_state(input_tensor.shape[0], tf.float32)
    for units in unit_size:
        cells.append(rnn.DropoutWrapper(rnn_cell(units),output_keep_prob=0.9))

    outputs = []
    tmp_tensor,state = input_tensor,initial_state

    for i in range(len_of_year):
        tmp_tensor,state = cells[i](tmp_tensor,state)
        outputs.append(tmp_tensor)
        tmp_tensor = tf.concat([input_tensor,tmp_tensor],axis=-1)
        tmp_tensor = layers.fully_connected(tmp_tensor,NUM_UNITS)

    outputs = tf.convert_to_tensor(outputs)

    # outputs, final_state = tf.nn.dynamic_rnn(
    #     cell=rnn_cell,  # 选择传入的cell
    #     inputs=net,  # 传入的数据
    #     initial_state=None,  # 初始状态
    #     dtype=tf.float32,  # 数据类型
    #     time_major=False,  # False: (batch, time_step, input); True: (time step, batch, input)，这里根据image结构选择False
    # )
    # # TODO: check the distribution

    output = tf.layers.dense(inputs=outputs, units=N_CLASSES, activation=tf.nn.tanh)
    tf.summary.histogram('output', output)
    # tf.summary.histogram('target', )
    # output: [batchsize,num_of_years,num_of_drugs]
    return output
