import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

hello = tf.constant("Hello TF!")

sess = tf.Session()

print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

tf.print(f"node1 : {node1}, node2 : {node2}")
tf.print(f"node3 : {node3}")

print(sess.run(node3))

ph1 = tf.placeholder(tf.float32)
ph2 = tf.placeholder(tf.float32)
add = ph1 + ph2


@tf.function
def forward():
    return ph1 + ph2


print(sess.run(add, feed_dict={ph1: 3, ph2: 4.5}))
print(sess.run(add, feed_dict={ph1: [1, 3], ph2: [2, 4]}))


