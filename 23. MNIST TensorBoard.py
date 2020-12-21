import datetime
import random
import ssl
import os

import matplotlib.pyplot as plt
import tensorflow as tf

random.seed(777)

ssl._create_default_https_context = ssl._create_unverified_context

mnist = tf.keras.datasets.mnist
print("mnist download complete")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("normalization done")

# linear classifier
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = os.path.join('.', "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train, y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

# test_output = model.predict(x_test)
#
# for _ in range(10):
#     r = random.randint(0, len(y_test))
#     label = y_test[r]
#     pred = tf.argmax(test_output[r])
#     print(f'Label : {label}')
#     print(f'Predict : {pred}')
#     plt.imshow(x_test[r].reshape(28, 28), cmap='Greys', interpolation='nearest')
#     plt.show()
