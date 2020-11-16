import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

# y=x 의 선형 그래프의 정점들
X = [1,2,3]
Y = [1,2,3]

# 처음엔 잘못된 기울기를 제공(정답은 1)
W = tf.Variable(-550.0)

# X * W 라는 가설
hypothesis = X * W

# cost(W)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 직접 미분하기 어려운 경우도 있으니, tensorflow의 함수를 이용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print(" step |           W ")
print("------|-------------")

for step in range(21):
  print("%5d | %12.9f"%(step, sess.run(W)))
  sess.run(train)