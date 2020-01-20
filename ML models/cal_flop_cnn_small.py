import tensorflow as tf
import tensorflow.contrib.slim as slim

g = tf.Graph()
run_meta = tf.RunMetadata()

with g.as_default():

    input_images = tf.placeholder("float", [1, 28, 28, 1])
    with slim.arg_scope([slim.conv2d], padding='VALID',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(input_images,10,[3,3],1,padding='SAME',scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net,64,[3,3],1,scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.conv2d(net,128,[3,3],1,scope='conv5')
            net = slim.flatten(net, scope='flat6')

            net = slim.fully_connected(net, 84, scope='fc7')
            digits = slim.fully_connected(net, 10, scope='fc9')

    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
    print('TF stats gives',flops.total_float_ops)
