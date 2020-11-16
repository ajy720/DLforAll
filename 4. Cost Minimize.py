import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

# y=x 의 선형 그래프의 정점들
X = [1,2,3]
Y = [1,2,3]

# 변수인 W는 값이 변할 수 있으므로 placeholder
W = tf.placeholder(tf.float32)

# X * W 라는 가설
hypothesis = X * W

# cost(W)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-30, 50):
  # W 값을 -3부터 5까지 변화시키면서, 그에 따른 cost(W)를 추출 
  feed_W = i * 0.1
  curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
  W_val.append(curr_W)
  cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()