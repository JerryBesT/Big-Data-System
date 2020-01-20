import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker": [
        "localhost:3333"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps": [
        "node0:3333"
    ],
    "worker": [
        "node0:3334",
        "node1:3333"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps": [
        "node0:3333"
    ],
    "worker": [
        "node0:3334",
        "node1:3333",
        "node2:3333",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)


class testHook(tf.train.FinalOpsHook):
    def __init__(self, final_ops, final_ops_feed_dict=None):
        self.final_ops = final_ops
        self.final_ops_feed_dict = final_ops_feed_dict

    def end(self, session):
        test_pred = session.run([self.final_ops], feed_dict=self.final_ops_feed_dict)[0]
        acc_num = np.sum(test_pred)
        print('acc: %d/%d' % (acc_num, mnist.test.num_examples))


if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    dsize = 784
    lsize = 10
    lr = 0.025
    epochs = 20
    batch_size = 10

    mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=clusterinfo)):

        X = tf.placeholder(tf.float32, [None, dsize])
        Y = tf.placeholder(tf.float32, [None, lsize])
        W = tf.Variable(tf.zeros([dsize, lsize]))
        b = tf.Variable(tf.zeros([lsize]))

        y_pred = tf.nn.softmax(tf.matmul(X, W) + b)
        loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_pred), reduction_indices=1))
        tf.summary.scalar("loss", loss)
        global_step = tf.train.get_or_create_global_step()
        corr_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))

        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)

        scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                     local_init_op=tf.local_variables_initializer())

        epoch_size = mnist.train.num_examples
        batch_num = epoch_size // batch_size
        hooks = [tf.train.StopAtStepHook(last_step=batch_num*epochs)]
        hooks.append(testHook(corr_pred, {X: mnist.test.images, Y: mnist.test.labels}))

    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="./logs/logistic_regression",
                                           hooks=hooks, scaffold=scaffold) as mon_sess:
        step = 0
        while not mon_sess.should_stop():
            cur_x, cur_y = mnist.train.next_batch(batch_size)
            _, l, step = mon_sess.run([optimizer, loss, global_step], feed_dict={X: cur_x, Y: cur_y})
            # print("Loss: ", l, "step: ", step)
