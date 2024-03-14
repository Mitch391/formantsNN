import settings
import numpy as np
import tensorflow as tf
from generators import sound_generator

# def resnet_sound_model(shape=(settings.conv_samples,settings.sound_sample_size)):
def resnet_sound_model(shape=(settings.sound_sample_size,settings.conv_samples)):
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.Normalization()(x_input)
    x = tf.keras.layers.GaussianNoise(0.2)(x)
    # x_original = tf.keras.layers.Flatten()(x)
    # x_original = tf.keras.layers.Dense(500)(x_original)
    # x_original = tf.keras.layers.Dense(192)(x_original)
    x = tf.keras.layers.Conv1D(256, settings.conv_kernel_size)(x)
    x = tf.keras.layers.Conv1D(settings.conv_filters_size, settings.conv_kernel_size)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.MaxPooling1D()(x)

    # # identity block
    # x_skip = x
    # x = tf.keras.layers.Conv1D(32, 3, padding = 'same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.Conv1D(32, 3, padding = 'same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Add()([x, x_skip])     
    # x = tf.keras.layers.Activation('relu')(x)

    # conv block
    for _ in range(3):
        x = conv_block(x)


    # # small conv model
    # x = tf.keras.layers.Conv1D(settings.conv_filters_size,settings.conv_kernel_size)(x)
    # # x = tf.keras.layers.Conv1D(settings.conv_filters_size,settings.conv_kernel_size)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.AveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # # x = tf.keras.layers.MaxPooling1D()(x)
    # x = tf.keras.layers.Conv1D(settings.conv_filters_size,settings.conv_kernel_size)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.AveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Conv1D(settings.conv_filters_size,settings.conv_kernel_size)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.AveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # # x = tf.keras.layers.MaxPooling1D()(x)
    # # x = tf.keras.layers.Conv1D(settings.conv_filters_size,settings.conv_kernel_size)(x)
    # # x = tf.keras.layers.AveragePooling1D()(x)

    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Add()([x_original, x])
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(5)(x)

    model = tf.keras.models.Model(inputs=x_input, outputs=x)
    return model

    
def conv_block(x):
    x_skip = x
    x = tf.keras.layers.Conv1D(settings.conv_filters_size,settings.conv_kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(settings.conv_filters_size,settings.conv_kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPooling1D()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    return x