import tensorflow as tf

n_vars = 28 * 28
n_layer1 = 300
n_layer2 = 100
n_outputs = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_vars), name='X')
y = tf.placeholder(tf.float32, shape=(None), name='y')

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
    trining_op = optimizer.minimize(loss)

with tf.name_scope("evaluation"):
    correct = tf.nn_in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("utils"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()