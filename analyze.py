import numpy as np
from model import build
from datasetup import setup

WEIGHTS = 'model.h5'

train_x_ims, train_x_text, train_y, test_x_ims, test_x_text, test_y, img_shape, vocab_size, n_ans, ans, test_q, test_answer_indices = setup()

model = build(img_shape, vocab_size, n_ans)
model.load_weights(WEIGHTS)
predictions = model.predict([test_x_ims, test_x_text])

for idx in range(n_ans):
    prediction = predictions[:, idx]
    answer = ans[idx]
    min = np.amin(prediction)
    max = np.amax(prediction)
    mean = np.mean(prediction)
    print("\n("+str(answer)+") Min: "+str(min)+", Max: "+str(max)+", Mean: "+str(mean))

shapes = []
decision = []

for i in range(n_ans):
    if ans[i] == 'rectangle' or ans[i] == 'circle' or ans[i] == 'triangle':
        shapes.append(i)
    elif ans[i] == 'yes' or ans[i] == 'no':
        decision.append(i)


def return_class(answer):
    if answer in shapes:
        return 0
    if answer in decision:
        return 1
    return 2


error_matrix = [[0 for _ in range(3)] for _ in range(3)]
total_errors = 0

color_error_matrix = [[0 for _ in range(n_ans)] for _ in range(n_ans)]
questions_wrong = 0

for idx in range(len(test_answer_indices)):
    ans = test_answer_indices[idx]
    pred = np.argmax(predictions[idx])
    if not ans == pred:
        total_errors += 1
        error_matrix[return_class(ans)][return_class(pred)] += 1
        color_error_matrix[ans][predictions] += 1
        if return_class(ans) == 1 and return_class(pred) == 1:
            if test_q[idx] in questions_wrong:
                questions_wrong[test_q[idx]] += 1
            else:
                questions_wrong[test_q[idx]] = 1

print('total error: '+str(total_errors / len(test_answer_indices)))
for i in range(3):
    print('{}\t{}\t{}\n'.format(error_matrix[i][0] / total_errors, error_matrix[i][1] / total_errors, error_matrix[i][2] / total_errors))
for i in range(n_ans):
  to_print = ''
  for j in range(n_ans):
    to_print += str(color_error_matrix[i][j]) + '\t'
  print(to_print)
print(questions_wrong)
