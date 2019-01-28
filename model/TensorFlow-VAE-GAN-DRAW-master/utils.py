import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from copy import deepcopy

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

def RNNdecoder_inference(input_tensor,output_tensor_last,N_CLASSES=69,NUM_UNITS=16,len_of_year = 8,mode="train"):
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
    batch_size = input_tensor.shape[0]
    tf.summary.histogram('concat', input_tensor)
    rnn_cell = rnn.GRUCell
    cells = []
    unit_size = [NUM_UNITS] * len_of_year
    for units in unit_size:
        cells.append(rnn.DropoutWrapper(rnn_cell(units), output_keep_prob=0.9))
    initial_state = cells[0].zero_state(batch_size=batch_size, dtype=tf.float32)
    outputs = []
    state = initial_state
    # net = tf.expand_dims(net, 1)
    output = tf.zeros([batch_size,N_CLASSES]) #b,69
    if mode=="test":
        for i in range(len_of_year):

            concated = tf.concat([input_tensor,output],axis=-1) # b, dim+69
            concated_ = tf.layers.dense(inputs=concated, units=NUM_UNITS, activation=tf.nn.tanh)  # b,dim+69 -> b,16
            output,state = cells[i](concated_,state)
            output = tf.layers.dense(inputs=output, units=N_CLASSES, activation=tf.nn.tanh)  # b,16 -> b,69
            outputs.append(output)
    else:
        # # z,y
        # # z:b,h
        # # y:b,69
        # # b,(h+69), hidden
        # # o:(b,16), hidden
        # # o - dense
        # y = 0 tensor
        # h = 0 tensor
        # for i in range(t):
        #     concat = tf.concat((z,y)) # b,69+dim
        #     concat_ = dense(concat) # b,dim2
        #     o,h = cell(concat_,h)
        #     y = dense(o)  # b,69

        # output_tensor_last = layers.fully_connected(output_tensor_last, NUM_UNITS) #?
        for i in range(len_of_year):
            concated = tf.concat([input_tensor,tf.squeeze(output_tensor_last[:,i,:])],axis=-1) #b,dim+69
            concated_ = tf.layers.dense(inputs=concated,units = NUM_UNITS, activation=tf.nn.tanh)#b,dim+69 -> b,16
            output,state = cells[i](concated_,state)
            output = tf.layers.dense(inputs=output, units=N_CLASSES, activation=tf.nn.tanh)#b,16 -> b,69
            outputs.append(output)
    outputs = tf.convert_to_tensor(outputs)#t,b,69
    outputs = tf.reshape(outputs,[batch_size,len_of_year,-1])#b,t,69
    tf.summary.histogram('outputs', outputs)

    # tf.summary.histogram('target', )
    # output: [batchsize,num_of_years,num_of_drugs]
    return outputs
