from keras.callbacks import ModelCheckpoint
from model import build
from datasetup import setup

train_x_ims, train_x_text, train_y, test_x_ims, test_x_text, test_y, img_shape, vocab_size, n_ans, _, _, _ = setup()

model = build(img_shape, vocab_size, n_ans)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

model.fit([train_x_ims, train_x_text], train_y, validation_data=([test_x_ims, test_x_text], test_y), shuffle=True, epochs=8, callbacks=[checkpoint])
