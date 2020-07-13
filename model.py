from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
from keras.optimizers import Adam


def build(img_shape, vocab_size, n_ans):
    img_input = Input(shape=img_shape)
    x1 = Conv2D(8, 3, padding='same')(img_input)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(16, 3, padding='same')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(32, 3, padding='same')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Flatten()(x1)
    x1 = Dense(32, activation='tanh')(x1)

    q_input = Input(shape=(vocab_size,))
    x2 = Dense(32, activation='tanh')(q_input)
    x2 = Dense(32, activation='tanh')(x2)

    out = Multiply()([x1, x2])
    out = Dense(32, activation='tanh')(out)
    out = Dense(n_ans, activation='softmax')(out)

    model = Model(inputs=[img_input, q_input], outputs=out)
    model.compile(Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

