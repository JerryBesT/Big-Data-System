from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import ModelBuilder
from .alexnetcommon import alexnet_inference, alexnet_part_conv, alexnet_loss, alexnet_eval
from ..optimizers.momentumhybrid import HybridMomentumOptimizer


def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
    """Build inference"""
    if devices is None:
        devices = [None]

    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    def train(total_loss, global_step, total_num_steps):
        """Build train operations"""
        # Compute gradients
        with tf.control_dependencies([total_loss]):
            opt = configure_optimizer(global_step, total_num_steps)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        with tf.control_dependencies([apply_gradient_op]):
            return tf.no_op(name='train')

    with tf.device(devices[0]):
        builder = ModelBuilder()
        print('num_classes: ' + str(num_classes))
        net, logits, total_loss = alexnet_inference(builder, images, labels, num_classes)

        if not is_train:
            return alexnet_eval(net, labels)

        global_step = builder.ensure_global_step()
        print('total_num_examples: ' + str(total_num_examples))
        train_op = train(total_loss, global_step, total_num_examples)
    return net, logits, total_loss, train_op, global_step


def distribute(images, labels, num_classes, total_num_examples, devices, is_train=True):
    # pass #placeholder

    # Put your code here
    # You can refer to the "original" function above, it is for the single-node version.
    # 1. Create global steps on the parameter server node. You can use the same method that the single-machine program uses.
    # 2. Configure your optimizer using HybridMomentumOptimizer.
    # 3. Construct graph replica by splitting the original tensors into sub tensors. (hint: take a look at tf.split )
    # 4. For each worker node, create replica by calling alexnet_inference and computing gradients.
    #    Reuse the variable for the next replica. For more information on how to reuse variables in TensorFlow,
    #    read how TensorFlow Variables work, and considering using tf.variable_scope.
    # 5. On the parameter server node, apply gradients.
    # 6. return required values.

    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    with tf.device(devices[-1]):
        print('num_classes: ' + str(num_classes))

        builder = ModelBuilder()
        # eval model
        if not is_train:
            net, logits, total_loss = alexnet_inference(builder, images, labels, num_classes)
            return alexnet_eval(net, labels)

        # distributed training
        print('total_num_examples: ' + str(total_num_examples))

        batch_size = images.get_shape().as_list()[0]
        print('batch_size: ' + str(batch_size))
        global_step = builder.ensure_global_step()
        opt = configure_optimizer(global_step, total_num_examples)

        num_workers = len(devices) - 1
        splits = []
        split_size = int(batch_size / num_workers)
        for i in range(num_workers - 1):
            splits.append(split_size)
        splits.append(batch_size - (num_workers - 1) * split_size)
        
        images_splits = tf.split(images, splits, name='split_images')
        labels_splits = tf.split(labels, splits, name='split_labels')
        
    grads_list = []
    net_list = []
    logits_list = []
    total_loss_list = []    

    for i in range(num_workers):
        with tf.variable_scope('graph', reuse = tf.AUTO_REUSE):
            with tf.device(devices[i]):
                net, logits, total_loss = alexnet_inference(builder, images_splits[i], labels_splits[i], num_classes)
                net_list.append(net)
                logits_list.append(logits)
                total_loss_list.append((float(splits[i]) / batch_size) * total_loss)
                curr_grads = opt.compute_gradients((float(splits[i]) / batch_size) * total_loss)
                grads_list.append(curr_grads)
   
    with tf.device(devices[-1]):
        grads = grads_list[0]
        for i in range(1,num_workers):
            grads += grads_list[i]

    	train_op = opt.apply_gradients(grads, global_step=global_step) 	  

    net = tf.concat(net_list, axis = 0)
    logits = tf.concat(logits_list, axis = 0)
    total_loss = tf.reduce_sum(total_loss_list)

    return net, logits, total_loss, train_op, global_step
