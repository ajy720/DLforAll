import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]


X = tf.placeholder("float", [None, 3]) # feature의 개수
Y = tf.placeholder("float", [None, 3]) # 클래스의 개수

W = tf.Variable(tf.random_normal([3, 3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')


logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# large_optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.5).minimize(cost)
# small_optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(cost)

prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer],
                            feed_dict={X: x_data, Y: y_data})
        print(step, cost_val)
        print(W_val)
    
    print("Predicton :", sess.run(prediction, feed_dict={X:x_test}))
    print("Accuracy :", sess.run(accuracy, feed_dict={X:x_test, Y:y_test}))
