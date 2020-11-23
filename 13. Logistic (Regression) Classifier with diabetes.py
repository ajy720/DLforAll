import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

xy = np.loadtxt('data/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sigmoid 함수를 이용한 가설, 0~1 사이의 실수가 나온다.
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

#  log 함수가 적용된 cost 함수
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# hypothesis가 0.5 보다 크면 True, 아니면 False인데, 이를 float32로 cast하면 1.0 혹은 0.0이 나온다.
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# 예측값과 Y값이 같은지 비교하고, 이 횟수를 통해 정확도(accuracy)를 구한다
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for step in range(15001):
    cost_val, _ = sess.run([cost, train],
                           feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
      print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})

  print(f'\nHypothsis: {h[-5:]}')
  print(f'\nCorrect (Y): {c[-5:]}')
  print(f'\nAccuracy: {a}')