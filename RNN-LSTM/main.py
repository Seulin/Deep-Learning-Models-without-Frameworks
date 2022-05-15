from data_train_test.emo_utils import *
import numpy as np
from model import *

# load data and glove
x_train, y_train = read_csv('./data_train_test/train_emoji.csv')
x_test, y_test = read_csv('./data_train_test/test_emoji.csv')
x_emojify, y_emojify = read_csv('./data_train_test/emojify_data.csv')

w2i, i2w, w2vm50 = read_glove_vecs('./glove.6B/glove.6B.50d.txt')
w2i, i2w, w2vm100 = read_glove_vecs('./glove.6B/glove.6B.100d.txt')

# data cleaning
b_size = 32

data_x = x_train[:106]
data_y = one_hot_encoder(y_train[:106])
dataset = list(dataloader(data_x, data_y, b_size))

vali = (x_train[110:], one_hot_encoder(y_train[110:]))

x_test = x_test
y_test = one_hot_encoder(y_test)


# a
epochs = 50
rnn = 0.0005
linear = 0.005
dim = 50
name = 'A'

model = Model('rnn', dim, 128, 2, 'sgd', [rnn, linear])
loss_func = CrossEntropyLoss()

train(model, loss_func, dataset, w2vm50, dim, vali, name, epochs)
test(model, x_test, y_test, w2vm50, dim)
draw_emojis(model, w2vm50, x_test, dim)

# b
epochs = 30
lstm = 0.00003
linear = 0.09
dim = 50
name = 'B'

model = Model('lstm', dim, 128, 2, 'sgd', [lstm, linear])
loss_func = CrossEntropyLoss()

train(model, loss_func, dataset, w2vm50, dim, vali, name, epochs)
test(model, x_test, y_test, w2vm50, dim)
draw_emojis(model, w2vm50, x_test, dim)

# c
epochs = 30
lstm = 0.00001
linear = 0.08
dim = 50
name = 'C'

model = Model('lstm', 50, 128, 2, 'adam', [lstm, linear])
loss_func = CrossEntropyLoss()

train(model, loss_func, dataset, w2vm50, dim, vali, name, epochs)
test(model, x_test, y_test, w2vm50, dim)
draw_emojis(model, w2vm50, x_test, dim)

# d
epochs = 30
lstm = 0.00003
linear = 0.04
dim = 100
name = 'D'

model = Model('lstm', dim, 128, 2, 'sgd', [lstm, linear])
loss_func = CrossEntropyLoss()

train(model, loss_func, dataset, w2vm100, dim, vali, name, epochs)
test(model, x_test, y_test, w2vm100, dim)
draw_emojis(model, w2vm100, x_test, dim)

# e
epochs = 30
lstm = 0.00001
linear = 0.09
dim = 50
name = 'E'

model = Model('lstm', dim, 128, 2, 'sgd', [lstm, linear], True)
loss_func = CrossEntropyLoss()

train(model, loss_func, dataset, w2vm50, dim, vali, name, epochs)
test(model, x_test, y_test, w2vm50, dim)
draw_emojis(model, w2vm50, x_test, dim)
