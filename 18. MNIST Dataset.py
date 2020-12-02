import matplotlib.pyplot as plt
import tensorflow as tf2
import ssl
import random

ssl._create_default_https_context = ssl._create_unverified_context

mnist = tf2.keras.datasets.mnist
print("mnist download complete")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
print("normalization done")

#linear classifier
model = tf2.keras.models.Sequential([
    tf2.keras.layers.Flatten(input_shape=(28,28)), #28 by 28 mnist input flatten
    tf2.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='SGD',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

for _ in range(10):
    r = random.randint(0, len(y_test))
    label = y_test[r:r+1]
    pred = tf2.argmax(model.predict(x_test[r:r+1]), 1)
    print(f'Label : {label}')
    print(f'Predict : {pred}')
    plt.imshow(x_test[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()