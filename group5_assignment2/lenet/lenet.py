from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os
import json

tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_integer("num_worker", 0, "Number of workers.")
FLAGS = tf.app.flags.FLAGS


NUM_WORKERS = FLAGS.num_worker

# configure the number of workers to realize distributed learning
if NUM_WORKERS == 1:
    workers = ["node0:2223"]
elif NUM_WORKERS == 2:
    workers = ["node0:2223", "node1:2222"]
else:
    workers = ["node0:2223", "node1:2222", "node2:2222"]

# in json form, configure the envrionment variable for the OS
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": workers,
    },
   "task": {"type": "worker", "index": FLAGS.task_index}
})


EPOCH = 20

datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

# repeat the number of dataset, which may not be enough for each worker to train
mnist_train, mnist_test = datasets['train'].repeat(EPOCH + 2), datasets['test']

# total counts of train and test dataset
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000
BATCH_SIZE = 64

# training format
def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label

# format the training and testing dataset
train_dataset_unbatched = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE)
train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset_unbatched = mnist_test.map(scale)

# according to the assignment spec, 3 convolutional layers (C1, C3 and C5), 
# 2 sub-sampling (pooling) layers (S2 and S4), and 1 fully connected layer (F6)
def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(6, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.AveragePooling2D(),
      tf.keras.layers.Conv2D(16, 3, activation='relu'),
      tf.keras.layers.AveragePooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(120, activation='relu'),
      tf.keras.layers.Dense(84, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  # Adam model, cross entropy loss function 
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      metrics=['accuracy'])
  return model

# enable multiple distributed learning in Keras
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


# Define the checkpoint directory to store the checkpoints
checkpoint_dir = '/users/huangrui/tf/checkpoint/lenet'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


# enable evaluation on tensor board to monitor the learning activity
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='/users/huangrui/tf/lenet_tb'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix)
]

# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64,
# and now this becomes 128.
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
train_dataset = train_dataset_unbatched.batch(GLOBAL_BATCH_SIZE)
with strategy.scope():
  multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x=train_dataset, epochs=EPOCH, steps_per_epoch=int(num_train_examples / GLOBAL_BATCH_SIZE), callbacks=callbacks)

# disable auto sharding
options = tf.data.Options()
options.experimental_distribute.auto_shard = False
eval_dataset = eval_dataset_unbatched.with_options(options).batch(GLOBAL_BATCH_SIZE)

eval_loss, eval_acc = multi_worker_model.evaluate(eval_dataset)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
