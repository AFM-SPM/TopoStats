import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


def classification_model(img_size, channels=1, classes=3, learning_rate=0.001):
    input_shape = (img_size, img_size, channels)
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the tensor output from the previous layer
    model.add(Flatten())

    # Dense layer to map the features to the output classes
    model.add(Dense(64, activation="relu"))

    # Dropout to reduce overfitting
    model.add(Dropout(0.5))

    # Output with 3 classes
    model.add(Dense(classes, activation="softmax"))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
