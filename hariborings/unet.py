"""A U-NET model for segmentation of Perovskite grains."""

from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.metrics import MeanIoU
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
    Lambda,
)
import tensorflow as tf

# Get IoU metric
# iou_loss = tf.keras.metrics.MeanIoU(num_classes=2)


def iou(y_true, y_pred):
    """Calculate the intersection over union loss."""
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (intersection + 1.0) / (K.sum(y_true_flat) + K.sum(y_pred_flat) - intersection + 1.0)


def iou_loss(y_true, y_pred):
    """Calculate the intersection over union loss."""
    return -iou(y_true, y_pred)


def unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, learning_rate: float = 0.01):
    """U-NET model definition function."""

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Downsampling
    # Downsample with increasing numbers of filters to try to capture more complex features (first argument)
    # Dropout is used to try to prevent overfitting. Increase if overfitting happens.
    # Dropout increases deeper into the model to further help prevent overfitting.

    conv1 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv1)
    pooled1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(pooled1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv2)
    pooled2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(pooled2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv3)
    pooled3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(pooled3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv4)
    pooled4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(pooled4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv5)

    # Upsampling
    # Conv2DTranspose is used as a sort of inverse convolution, to upsample the image
    # A concatenation is used to force context from the original image, providing information about what context a
    # feature stems from.

    up6 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv6)

    up7 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv7)

    up8 = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(up8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv8)

    up9 = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv9)

    # Make predictions of classes based on the culminated data
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # sgd = SGD(learning_rate=0.01)
    # For images with a lot of background, try using a weighted loss function to help the model focus on the grains
    # model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["accuracy"], sample_weight_mode="temporal")
    # For images with a lot of background, try a metric that focuses on the grains
    # model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["accuracy", "mean_squared_error"])
    # For images with a lot of background, try a metric that focuses on the grains such as intersection over union
    # model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=[iou_loss])
    # Try standard binary crossentropy with standard accuracy
    # model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["accuracy"])

    optimiser = Adam(learning_rate)
    # model.compile(optimizer=optimiser, loss="binary_crossentropy", metrics=["accuracy"])
    # model.compile(
    #     optimizer=optimiser,
    #     loss="binary_crossentropy",
    #     metrics=[MeanIoU(num_classes=2), "accuracy"],
    # )

    # IOU
    model.compile(optimizer=optimiser, loss=iou_loss, metrics=[iou, "accuracy"])

    model.summary()

    return model