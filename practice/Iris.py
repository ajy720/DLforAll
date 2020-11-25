# 붓꽃의 꽃받침과 꽃잎의 너비와 길이 데이터를 학습해,
# setosa, versicolor, virginica 중 어떤 종인지 예측
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

iris_dict = {
  0 : 'Iris-setosa',
  1 : 'Iris-versicolor',
  2 : 'Iris-virginica'
}

xy = np.loadtxt('./practice/Iris.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


nb_classes = 3 # setosa(0), versicolor(1), virginica(2)

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)                                              
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())


  for step in range(10001):
    sess.run(optimizer, feed_dict={X: x_data, Y: y_data})

    if step % 100 == 0:
      loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
      print("Step: {:4}\tLoss: {:3.6f}\tAcc: {:3.6%}".format(step, loss, acc))
  
  pred = sess.run(prediction, feed_dict={X: x_data})

  for p, y in zip(pred, y_data.flatten()):
    print(f"[{p==y}]\tPred: {p}\tReal Y: {y}")

  print(f"{len(*pred[True])}/{len(pred)}")

