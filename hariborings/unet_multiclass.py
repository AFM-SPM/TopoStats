"""A U-NET model for segmentation of Perovskite grains."""

from keras.models import Model
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
from keras.optimizers import Adam
import tensorflow as tf

num_classes = 3

# def mean_iou(y_true, y_pred):
#     """Mean Intersection Over Union metric."""
#     y_pred = tf.round(tf.cast(y_pred, tf.int32))
#     intersect = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32), axis=[1])
#     union = tf.reduce_sum(tf.cast(y_true, tf.float32), axis=[1]) + tf.reduce_sum(
#         tf.cast(y_pred, tf.float32), axis=[1]
#     )
#     smooth = tf.ones(tf.shape(intersect))
#     return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))


def mean_iou(y_true, y_pred):
    """Mean Intersection Over Union metric, ignoring the background class."""
    y_true_f = tf.reshape(y_true[:, :, :, 1:], [-1])  # ignore background class
    y_pred_f = tf.round(tf.reshape(y_pred[:, :, :, 1:], [-1]))  # ignore background class
    intersect = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersect
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))


# def iou_loss(y_true, y_pred):
#     """Intersection Over Union loss function."""
#     y_true = tf.reshape(y_true, [-1])
#     y_pred = tf.reshape(y_pred, [-1])
#     intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32))
#     score = (intersection + 1.0) / (
#         tf.reduce_sum(tf.cast(y_true, tf.float32))
#         + tf.reduce_sum(tf.cast(y_pred, tf.float32))
#         - intersection
#         + 1.0
#     )
#     return 1.0 - score


def iou_loss(y_true, y_pred):
    """IoU Loss for 2 tensors, ignoring the background class."""
    y_true_f = tf.reshape(y_true[:, :, :, 1:], [-1])  # ignore background class
    y_pred_f = tf.reshape(y_pred[:, :, :, 1:], [-1])  # ignore background class
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return 1 - ((intersection + 1.0) / (union + 1.0))


# def dice_coefficient(y_true, y_pred):
#     """Dice Coefficient for 2 tensors."""
#     smooth = 1.0
#     y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
#     y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     return (2.0 * intersection + smooth) / (
#         tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
#     )


# IGNORE BACKGROUND CLASS
def dice_coefficient(y_true, y_pred):
    """Dice Coefficient for 2 tensors, ignoring the background class."""
    # y_true_f = tf.reshape(y_true[:, :, :, 1:], [-1])  # ignore background class
    # y_pred_f = tf.reshape(y_pred[:, :, :, 1:], [-1])  # ignore background class
    # don't ignore background class
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0)


def dice_loss(y_true, y_pred):
    """Dice Loss for 2 tensors."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def dice_accuracy(y_true, y_pred):
    """Dice accuracy for 2 tensors."""
    return dice_coefficient(y_true, y_pred)


def multiclass_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, learning_rate=0.001):
    """U-NET model definition function."""

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # inputs = Input(shape=(None, None, IMG_CHANNELS))

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
    outputs = Conv2D(num_classes, kernel_size=(1, 1), activation="softmax")(conv9)

    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", mean_iou])
    # model.compile(optimizer=optimizer, loss=iou_loss, metrics=["accuracy", mean_iou])
    # model.compile(optimizer=optimizer, loss=dice_loss, metrics=["accuracy", dice_accuracy])
    model.summary()

    return model
