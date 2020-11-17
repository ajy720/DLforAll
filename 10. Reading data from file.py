import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
tf.disable_v2_behavior()

xy = np.loadtxt('./data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# print(x_data.shape, x_data, len(x_data))
# print(y_data.shape, y_data)


X = tf.placeholder(tf.float32, shape=[None, 3])

Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for step in range(100001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                feed_dict={X: x_data, Y: y_data})

    if step % 1000 == 0:
        print(f'{step} Cost : {cost_val}')            
        print(f'Prediction {hy_val}\n')            
        
