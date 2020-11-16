import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

# y=x 의 선형 그래프의 정점들
x_data = [1,2,3]
y_data = [1,2,3]

# 식에 대입되는 값들
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# X * W 라는 가설
hypothesis = X * W

# cost(W)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# learning_rate로 학습 범위 지정(너무 휙휙 움직이면 안 되니까, 정확한 값을 위해)
learning_rate = 0.1
# Gradient를 정의
gradient = tf.reduce_mean((W * X - Y) * X)
# Gradient에 따른 하강 정도
descent = W - learning_rate * gradient
# update에 Gradient Descent 할당
update = W.assign(descent)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print(" step |        cost |           W ")
print("------|-------------|-------------")

for step in range(21):
  sess.run(update, feed_dict={X: x_data, Y: y_data})
  print("%5d | %11.9f | %12.9f"
    %(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)))