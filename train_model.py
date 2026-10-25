import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10)
])

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training CNN...")

model.fit(
    x_train,
    y_train,
    epochs=8,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nCNN MODEL RESULTS")
print("------------------")
print("Test Accuracy:", round(accuracy * 100, 2), "%")

model.save("digit_model.keras")

print("\nModel saved successfully!")
print("File: digit_model.keras")
print("\nTraining is complete.")
print("You do NOT need to train the model again.")