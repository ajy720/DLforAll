import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
tf.disable_v2_behavior()


# 파일 큐 생성
filename_queue = tf.train.string_input_producer(
    ['./data/data-01-test-score.csv'], shuffle=False, name='filename_queue')

# 리더 정의
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 레코드 기본값
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# x, y 의 배치를 구성(어디서, 몇개씩 가져올 것인지)
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)


X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# filename Queue를 관리해주는 통상적인 부분
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


for step in range(100001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])

    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                feed_dict={X: x_batch, Y: y_batch})

    if step % 1000 == 0:
        print(f'{step} Cost : {cost_val}')            
        print(f'Prediction {hy_val}\n')            
        

coord.request_stop()
coord.join(threads)