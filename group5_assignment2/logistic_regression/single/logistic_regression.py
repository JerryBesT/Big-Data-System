import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

# A tf.train.ClusterSpec represents the set of processes that 
# participate in a distributed TensorFlow computation. 
# Every tf.distribute.Server is constructed in a particular cluster.
clusterSpec_single = tf.train.ClusterSpec({
    "worker": [
        "localhost:2222"
    ]
})

# each cluster defines the "chief" node as parameter and the worker nodes
clusterSpec_cluster = tf.train.ClusterSpec({
    "ps": [
        "node0:2222"
    ],
    "worker": [
        "node0:2223",
        "node1:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps": [
        "node0:2222"
    ],
    "worker": [
        "node0:2223",
        "node1:2222",
        "node2:2222",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

workerNum = {
    "single": 1,
    "cluster": 2,
    "cluster2": 3
}

# implement testhook to optimize the calculation - lazy batch
class testHook(tf.train.FinalOpsHook):
    def __init__(self,final_ops,final_ops_feed_dict=None):
        self.final_ops = final_ops
        self.final_ops_feed_dict = final_ops_feed_dict

    # operation at the end of each session, test accuracy
    def end(self,session):
        test_pred = session.run([self.final_ops], feed_dict=self.final_ops_feed_dict)[0]
        acc_num = np.sum(test_pred)
        print('acc: %d/%d' % (acc_num, mnist.test.num_examples))

# contains the info whether the training is done on single or distributed machines.
clusterinfo = clusterSpec[FLAGS.deploy_mode]

# initialize the server for the cluster to launch, maybe single or distributed
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

# if the given flag is master, then keep waiting
if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # training data size, label size, learning rate, epoch nums, batch size for each SGD
    dsize = 784
    lsize = 10
    lr = 0.025
    epochs = 20
    batch_size = 128

    mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

    # for each worker, start training
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=clusterinfo)):

        # the feeding vectors for graph input
        X = tf.placeholder(tf.float32, [None, dsize])
        Y = tf.placeholder(tf.float32, [None, lsize])

        # the trained weight and regularization biase
        W = tf.Variable(tf.zeros([dsize, lsize]))
        b = tf.Variable(tf.zeros([lsize]))

        # Construct model
        y_pred = tf.nn.softmax(tf.matmul(X, W) + b)

        # Minimize error using cross entropy
        loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_pred), reduction_indices=1))

        # for tensorboard
        tf.summary.scalar("train/loss", loss)

        # global counter for batches / second
        global_step = tf.contrib.framework.get_or_create_global_step()

        # check prediction with corresponding label
        corr_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))

        # accuracy for all the training
        train_acc = tf.reduce_sum(tf.cast(corr_pred, tf.int32)) / batch_size

        # for tensorboard
        tf.summary.scalar("train/acc", train_acc)

        # total of epochs and batch for each epoch
        epoch_size = mnist.train.num_examples
        batch_num = epoch_size // batch_size
        hooks = [tf.train.StopAtStepHook(last_step=batch_num*epochs)]

        # learning by modifying weights according to the SGD
        optimizer = tf.train.GradientDescentOptimizer(lr)

        # sycnchronize the updates for the main updates.
        optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=workerNum[FLAGS.deploy_mode],
                                                   total_num_replicas=workerNum[FLAGS.deploy_mode])
        # launch the global step
        operation = optimizer.minimize(loss, global_step=global_step)
        hooks.append(optimizer.make_session_run_hook((FLAGS.task_index==0)))
        hooks.append(testHook(corr_pred,{X: mnist.test.images, Y: mnist.test.labels}))

        scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                     local_init_op=tf.local_variables_initializer())

        # to monitor the training session, with manual set hooks, scaffold
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="./checkpoint/lr_single",
                                               scaffold = scaffold,
                                               hooks=hooks,
                                               ) as mon_sess:
            # start training
            while not mon_sess.should_stop():
                # start the next batch, with the input, label
                cur_x, cur_y = mnist.train.next_batch(batch_size)
                _, l, step = mon_sess.run([operation, loss, global_step], feed_dict={X: cur_x, Y: cur_y})