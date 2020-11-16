import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

# y=x 의 선형 그래프의 정점들
X = [1,2,3]
Y = [1,2,3]

# 처음엔 잘못된 기울기를 제공(정답은 1)
W = tf.Variable(5.0)


hypothesis = X * W

# 직접 미분한 경사
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# tensorflow에서 제공하는 기능(cost/loss function)
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# gradient 계산
gvs = optimizer.compute_gradients(cost)

# gradient 적용
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print(" step, gradient, W, gvs ")

for step in range(101):
  print(step, sess.run([gradient, W, gvs]))
  sess.run(apply_gradients)