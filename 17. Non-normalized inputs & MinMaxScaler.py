import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

def MinMaxScaler(data):
    # 1. 시작 범위를 0으로 만든다.
    # 2. 원소 중 최대값으로 나눠주어 최대값이 1이 되게 한다.

    # 최소값으로 빼줘서 범위가 0~?이 되게 함
    numerate = data - np.min(data)

    # 변한 범위의 최대값
    denominator = np.max(data) - np.min(data)
    
    # 약간의 텀을 더해줘서 0으로 나뉘어지는걸 방지
    return numerate / (denominator + 1e-7)


xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

xy = MinMaxScaler(xy)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]



X = tf.placeholder("float", [None, 4]) # feature의 개수
Y = tf.placeholder("float", [None, 1]) # 클래스의 개수

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                feed_dict={X: x_data, Y: y_data})

    if step % 10 == 0:
        print(f'{step} Cost : {cost_val}')            
        print(f'Prediction {hy_val}\n')            
        





