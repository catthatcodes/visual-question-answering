from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from easy_vqa import get_train_questions, get_test_questions, get_train_image_paths, get_test_image_paths, get_answers


def preprocess_images(path):
    img = img_to_array(load_img(path))
    return img / 255 - 0.5


def read_images(path):
    imgs = {}
    for img_id, img_path in path.items():
        imgs[img_id] = preprocess_images(img_path)
    return imgs


def setup():
    train_q, train_a, train_id = get_train_questions()
    test_q, test_a, test_id = get_test_questions()
    print("Training questions = " + len(train_q) + "\nTesting questions =" + len(test_q))

    ans = get_answers()
    n_ans = len(ans)
    print("Answers = " + n_ans)

    train_imgs = read_images(get_train_image_paths())
    test_imgs = read_images(get_test_image_paths())
    img_shape = train_imgs[0].shape
    print("Each image has shape: " + img_shape)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_q)
    vocab_size = len(tokenizer.word_index) + 1

    train_x_text = tokenizer.texts_to_matrix(train_q)
    test_x_text = tokenizer.texts_to_matrix(test_q)
    train_x_ims = [train_imgs[id] for id in train_id]
    test_x_ims = [test_imgs[id] for id in test_id]

    train_answer_indices = [ans.index(a) for a in train_a]
    test_answer_indices = [ans.index(a) for a in test_a]
    train_y = to_categorical(train_answer_indices)
    test_y = to_categorical(test_answer_indices)

    return (train_x_ims, train_x_text, train_y, test_x_ims, test_x_text, test_y, img_shape, vocab_size, n_ans, ans, test_q, test_answer_indices)