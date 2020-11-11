import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

x_train = [1, 2, 3]
y_train = [1, 2, 3]

# random_normal 안에 tensor shape를 준다 (1차원, 원소는 1개 = 숫자 하나)
W = tf.Variable(tf.random_normal([1]), name="weight")

b = tf.Variable(tf.random_normal([1]), name="bias")

# H(x) = Wx+b
hypothesis = x_train * W + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print(" step |      cost |         W |         b")
print("------|-----------|-----------|----------")
for step in range(5001):
    sess.run(train)
    if step % 20 == 0:
        print(
            "%5d | %2.7f | %2.7f | %2.7f"
            % (step, sess.run(cost), sess.run(W), sess.run(b))
        )
