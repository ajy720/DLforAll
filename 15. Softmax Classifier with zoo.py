import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
        
nb_classes = 7



X = tf.placeholder(tf.float32, [None, 16]) # feature의 개수
Y = tf.placeholder(tf.int32, [None, 1]) # 입력 들어오는 데이터는 1자리 수 정수로 들어온다. shape = (None, 1)

Y_one_hot = tf.one_hot(Y, nb_classes) # 이를 one-hot으로 만들어준다. 
# 하지만 ont_hot(Y, bm_classes)은 기존 shape이 (n, 1)이라면 (n, 1, nb_classes)의 형태로 만들어 반환하기 때문에,
# 이를 (None, nb_classes)로 간편하게 사용하기 위해 다시 reshape 해줘야 한다.

Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# None처럼 정해지지 않은 크기는 reshape에서 -1로 사용한다

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
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

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
  sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
  if step % 100 == 0:
    loss, acc = sess.run([cost, accuracy], 
      feed_dict={X: x_data, Y: y_data})

    print("Step: {:5}\t Loss: {:3.3f}\t Acc: {:3.2%}".format(step, loss, acc))


pred = sess.run(prediction, feed_dict={X: x_data})

for p, y in zip(pred, y_data.flatten()):
  print("[{}]\t Prediction: {}\t True Y: {}".format(p==int(y), p, int(y)))



