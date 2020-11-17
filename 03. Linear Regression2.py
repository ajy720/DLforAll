import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# random_normal 안에 tensor shape를 준다 (1차원, 원소는 1개 = 숫자 하나)
W = tf.Variable(tf.random_normal([1]), name="weight")

b = tf.Variable(tf.random_normal([1]), name="bias")

# H(x) = Wx+b
hypothesis = X * W + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print(" step |        cost |           W |           b")
print("------|-------------|-------------|------------")
for step in range(3001):
    cost_val, w_val, b_val, _ = sess.run(
        [cost, W, b, train],
        feed_dict={
          X: [1, 2, 3, 4, 5],
          Y: [2.1, 3.1, 4.1, 5.1, 6.1]},
    )

    if step % 100 == 0:
        print("%5d | %11.9f | %12.9f | %12.9f" 
        % (step, cost_val, w_val, b_val))


# testing out model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [-13, 36]}))
print(sess.run(hypothesis, feed_dict={X: [-34.1, 23.5]}))