import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0],    [1],    [1],    [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

width = 5
depth = 3

W = []
b = []
layer = []


W.append(tf.Variable(tf.random_normal([2, width]), name='weight0'))
b.append(tf.Variable(tf.random_normal([width]), name='bias0'))
layer.append(tf.sigmoid(tf.matmul(X, W[0]) + b[0]))

for i in range(1, width-1):
    W.append(tf.Variable(tf.random_normal([width, width]), name=f'weight{i}'))
    b.append(tf.Variable(tf.random_normal([width]), name=f'bias{i}'))
    layer.append(tf.sigmoid(tf.matmul(layer[-1], W[i]) + b[i]))

W.append(tf.Variable(tf.random_normal([width, 1]), name=f'weight{width-1}'))
b.append(tf.Variable(tf.random_normal([1]), name=f'bias{width-1}'))
hypothesis = tf.sigmoid(tf.matmul(layer[-1], W[-1]) + b[-1])



cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(f"{step} {sess.run(cost, feed_dict={X: x_data, Y: y_data})} \n{sess.run(W)}")

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print()
    print("Hypothesis :")
    print(h)
    print("Correct")
    print(c)
    print("Accuracy")
    print(a)
