import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4]) # feature의 개수
Y = tf.placeholder("float", [None, 3]) # 클래스의 개수
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')


logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# 직접 수식을 작성해서 하는 방식
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# -----

# TF에서 지원하는 softmax_cross_entropy_with_logits라는 함수를 이용해 하는 방식
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y)

cost = tf.reduce_mean(cost_i)   
# -----                                        

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
  sess.run(optimizer, feed_dict={X: x_data, Y:y_data})
  if step % 200 == 0:
    print(step, sess.run(cost, feed_dict={X: x_data, Y:y_data}))


# arg_max 메서드로 softmax를 통해 얻어낸 결과값 중 가장 유력한 후보만 hot-encoding 할 수 있다
a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
print(a, sess.run(tf.arg_max(a, 1)))

all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9],
                                          [1, 3, 4, 3],
                                          [1, 1, 0, 1]]})
print(all, sess.run(tf.arg_max(all, 1)))