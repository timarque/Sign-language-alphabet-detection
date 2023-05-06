from tensorflow.keras.layers import BatchNormalization, Conv2D, Concatenate, Dense, Dropout, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model


def dense_block(x, layers):
    for i in range(layers):
        x_n = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x_n = BatchNormalization()(x_n)
        x = Concatenate()([x, x_n])
    return x


def transition_layer(x):
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x


def DenseNet(input_shape, num_classes):
    input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = dense_block(x, 10)
    x = transition_layer(x)

    x = dense_block(x, 10)
    x = transition_layer(x)

    x = dense_block(x, 10)

    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)
    return model
