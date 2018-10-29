import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_vars = 28 * 28
n_layer1 = 300
n_layer2 = 100
n_outputs = 10
learning_rate = 0.01

n_epochs = 40
batch_size = 50

X = tf.placeholder(tf.float32, shape=(None, n_vars), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')

def add_layer(X, n_units, name, activation=None):
    with tf.name_scope(name):
        input_size = int(X.get_shape()[1])
        sigma = 2 / (input_size + n_units)
        weights = tf.truncated_normal((input_size, n_units), stddev=sigma)
        W = tf.Variable(weights, name='W')
        b = tf.Variable(tf.zeros([n_units]), name = 'b')
        output = tf.matmul(X, W) + b
        if activation is not None:
            return activation(output)
        else:
            return output

with tf.name_scope("mlp"):
    layer1 = add_layer(X, 300, 'layer1_output', activation=tf.nn.relu)
    layer2 = add_layer(layer1, 100, 'layer2_output', activation=tf.nn.relu)
    logits = add_layer(layer2, 10, 'logits')

with tf.name_scope("loss"):
    cross_entropy_result = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy_result)

with tf.name_scope("optimization"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("evaluation"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("utils"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

if __name__ == '__main__':
    mnist = input_data.read_data_sets("/tmp/data/")

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            if epoch % 10 == 0:
                last_check_point_path = saver.save(sess, './checkpoints/checkpoint_{}'.format(epoch))
                print('Checkpoint created')
            print("[{}] accuracy: {}".format(epoch, acc_train))
        save_path = saver.save(sess, 'models/model.ckpt')
        print('Model saved at {}'.format(save_path))