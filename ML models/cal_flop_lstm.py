import tensorflow as tf
from tensorflow.contrib import rnn

#import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

g = tf.Graph()
run_meta = tf.RunMetadata()

with g.as_default():

	time_steps=28
	#hidden LSTM units
	num_units=128
	#rows of 28 pixels
	n_input=28
	#learning rate for adam
	learning_rate=0.001
	#mnist is meant to be classified in 10 classes(0-9).
	n_classes=10
	#size of batch
	batch_size=128

	#defining placeholders
	#input image placeholder
	x=tf.placeholder("float",[1,time_steps,n_input])

	out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
	out_bias=tf.Variable(tf.random_normal([n_classes]))

	#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
	input=tf.unstack(x ,time_steps,1)

	#defining the network
	lstm_layer=tf.contrib.rnn.LSTMCell(num_units)
	outputs, _=tf.nn.static_rnn(lstm_layer,input,dtype="float32")
	
	prediction=tf.matmul(outputs[-1],out_weights)+out_bias


	opts = tf.profiler.ProfileOptionBuilder.float_operation()
	flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
	print('TF stats gives',flops.total_float_ops)
