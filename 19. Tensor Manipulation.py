import tensorflow as tf
import numpy as np
import pprint

# tf.compat.v1.disable_v2_behavior()

pp = pprint.PrettyPrinter(indent=4)

t = np.array([0., 1., 2., 3., 4., 5., 6.])


pp.pprint(t)
print(t.ndim) # rank - n차원 배열
print(t.shape) # shape - 배열의 모양

t = tf.constant([1, 2, 3, 4])
print(tf.shape(t))

t = tf.constant([[1, 2],
                 [3, 4]])
print(tf.shape(t))

t = tf.constant(
    [[[
        [ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]
    ],[
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ]]]
)
print(tf.shape(t))

# Shape, Rank, Axis

# rank = axis의 갯수
# shape = 각 axis의 너비(길이)
# axis = 축
# axis 번호는 가장 바깥에서 0부터 센다


matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print(f"Matrix 1 shape : {matrix1.shape}")
print(f"Matrix 2 shape : {matrix2.shape}")

tf.matmul(matrix1, matrix2)
print(matrix1 * matrix2)
# 그냥 곱하기 해버리면 우리가 기대한 행렬곱의 결과와는 차이가 생기기 때문에 유의.

# shape이 다른데도 불구하고 그냥 곱하기가 가능한 이유는?
# Broadcasting

# shape가 같다면 1:1로 매핑해서 연산
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
print(matrix1 + matrix2)

# shape가 달라도 자동으로 변환해서 연산하게 해주는 기능이 Broadcasting
# 기능 자체는 매우 강력하지만, 의도하지 않은 결과를 얻을 수 있기 때문에 유의해서 사용.
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant(3.)
print(matrix1 + matrix2)

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([[3., 4.]])
print(matrix1 + matrix2)

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([[3.], [4.]])
print(matrix1 + matrix2)


# Reduce Family

# floating point 조심
tf.reduce_mean([1, 2], axis=0)

x = [[1., 2.],
     [3., 4.]]
tf.reduce_mean(x)
tf.reduce_sum(x)

# axis를 설정해서 원하는 방향의 reduce 연산이 가능
# -1(제일 안쪽 축)은 보통 안쪽의 합을 먼저 구하기 위해 사용
tf.reduce_mean(x, axis=0)
tf.reduce_mean(x, axis=1)
tf.reduce_mean(x, axis=-1)

tf.reduce_sum(x, axis=0)
tf.reduce_sum(x, axis=1)
tf.reduce_sum(x, axis=-1)


# Argmax
# Argmax에서도 똑같이 axis 사용

x = [[0, 1, 2],
     [2, 1, 0]]

tf.argmax(x, axis=0)  # -> [1, 0, 0]
tf.argmax(x, axis=1)  # -> [2, 0]
tf.argmax(x, axis=-1) # -> [2, 0]


# Reshape ⭐⭐
t = np.array([[[ 0,  1,  2],
               [ 3,  4,  5]],
       
              [[ 6,  7,  8],
               [ 9, 10, 11]]])
t.shape

# 보통 reshape 할 때 가장 안쪽의 모양은 그대로 가져간다
# -1은 그 앞에건 알아서 하라는 의미
tf.reshape(t, shape=[-1, 3])

# 사이에 1을 하나 끼워서 한겹 더 두른 형태로 만들 수도 있다
tf.reshape(t, shape=[-1, 1, 3])


# Squeeze
# 모든 요소에 똑같이 n개씩 껴있는 쓸모없는 축을 쥐어 짜내는 것
tf.squeeze([[0], [1], [2]]) # -> [0, 1, 2]

tf.squeeze([[[ 0,  1,  2]],  # -> [[ 0,  1,  2],
            [[ 3,  4,  5]],  # ->  [ 3,  4,  5],
            [[ 6,  7,  8]],  # ->  [ 6,  7,  8],
            [[ 9, 10, 11]]]) # ->  [ 9, 10, 11]]


# Expand
# squeeze와 정반대의 기능. 모든 요소에 축을 n개씩 더 끼워 넣는다
tf.expand_dims([0, 1, 2], 1) # -> [[0], [1], [2]]

tf.expand_dims([[ 0,  1,  2],     # -> [[[ 0,  1,  2]],
                [ 3,  4,  5],     # ->  [[ 3,  4,  5]],
                [ 6,  7,  8],     # ->  [[ 6,  7,  8]],
                [ 9, 10, 11]], 1) # ->  [[ 9, 10, 11]]]


# One hot
# 깊이를 알려주고 어떤 배열을 던져주면 
# 각 요소를 One-hot encoding한 배열로 변환해 반환한다
# 단, 자동으로 한번 expand가 되어 나오기 때문에 기호에 따라 reshape를 사용한다.
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3])


# Casting
# python 기본 map 처럼 사용할 수 있음
# 각 요소를 특정 자료형으로 변환해준다.
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32) # -> [1, 2, 3, 4]

# Boolean 값을 변환해 Accuracy를 구할 때 사용하기도 한다
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32) # -> [1, 0, 1, 0]


# Stack
# 배열을 쌓아준다. axis로 축을 지정해 원하는 방향으로 쌓는 것도 가능하다.
x = [1, 4]
y = [2, 5]
z = [3, 6]

tf.stack([x, y, z])
# ->
# [[1, 4],
#  [2, 5],
#  [3, 6]]

tf.stack([x, y, z], axis=1)
# ->
# [[1, 2, 3],
#  [4, 5, 6]]


# Ones and Zeros like
# 어떤 배열을 받으면 1 혹은 0으로 채워진 그 배열과 같은 shape를 가진 배열 생성
x = [[0, 1, 2],
     [2, 1, 0]]

tf.ones_like(x)
# ->
# [[1, 1, 1],
#  [1, 1, 1]]
tf.zeros_like(x)
# ->
# [[0, 0, 0],
#  [0, 0, 0]]

# Zip
# python 내장 함수, 기존 사용법과 같다