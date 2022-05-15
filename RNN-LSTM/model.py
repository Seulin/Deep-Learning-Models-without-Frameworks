from rnn import *
from lstm import *
from module import *
import matplotlib.pyplot as plt
from data_train_test.emo_utils import *

def one_hot_encoder(data):
    encoded = np.array([[1 if label == i else 0 for i in range(5)] for label in data])
    return encoded

def one_hot_decoder(data):
    decoded = np.argmax(data, axis = 1)
    return decoded

class Model(baseLayer):
    def __init__(self, model, i_size, h_size, l_size, optim, lr = None, dropout = False):

        if lr == None:
            lr = [0.001, 0.001]
        if model == 'rnn':
            rnn = RNN(i_size, h_size, l_size, lr[0])
            self.layers = [rnn]
        elif model == 'lstm':
            lstm = LSTM(i_size, h_size, l_size, optim, lr[0])
            self.layers = [lstm]
        if dropout:
            self.layers.append(Dropout())

        linear = Linear(h_size, 5, lr[1]) # 감정 5개
        soft = Softmax()
        self.layers.extend([linear, soft])

    def forward(self, input):
        data = input
        for i, layer in enumerate(self.layers):
            data = layer(data)
        return data

    def backward(self, data): # including update
        for i, layer in enumerate(reversed(self.layers)):
            data = layer.backward(data)

    def predict(self, x):
        prob = self.forward(x)
        return one_hot_decoder(prob)

    def score(self, x, y):
        correct = wrong = 0
        predicted = self.predict(x)
        for label, pred in zip(y, predicted):
            if (label[pred] == 1):
                correct += 1
            else:
                wrong += 1
        return correct / (correct+wrong)


def embedding(w2vm, input, dim):
    res = np.zeros((len(input), 10, dim)) # 10 == w_size, 50=i_size
    for i, sen in enumerate(input):
        words = sen.split()
        for j, word in enumerate(words):
            res[i][j] = w2vm[word.lower()]
    return res.transpose(1, 0, 2)

def dataloader(x, y, b_size): # x[1], y[0] = batch dimension
    num_batch, remainder = divmod(x.shape[0], b_size)
    remainder = 1 if remainder != 0 else 0

    for i in range(num_batch + remainder):
        start = i*b_size
        if i == num_batch + remainder - 1:
            yield [ x[start:], y[start:] ]
        else:
            yield  [ x[start:start + b_size], y[start:start + b_size] ]

def test(model, x, y, w2vm, dim):
    x = embedding(w2vm, x, dim)
    accuracy = model.score(x, y)
    print('--Accuracy--')
    print(accuracy)
    return accuracy

def train(model, loss_func, dataset, w2vm, dim, validation, name, epochs):
    print('--Loss--')
#     accuracy = []
    history = {'t_loss': [],'v_loss': [], 't_accu':[], 'v_accu':[]} # train_loss, validation_loss
    for epoch in range(epochs):
        total_loss = 0
        total = 0
        for x, y in dataset:
            x = embedding(w2vm, x, dim)
            output = model(x)
            loss = loss_func(output, y)
            loss_grad = loss_func.backward()
            model.backward(loss_grad)

            total += x.shape[1]
            total_loss += np.sum(loss)

        # validation_loss
        output = model(embedding(w2vm, validation[0], dim))
        loss = loss_func(output, validation[1])

        history['t_loss'].append(total_loss/total)
        history['v_loss'].append(np.mean(loss))

        v_accu = model.score(embedding(w2vm, validation[0], dim), validation[1])
        history['t_accu'].append(model.score(x, y))
        history['v_accu'].append(v_accu)

#         accuracy.append(accu)
        print(f'{epoch + 1} epoch: {total_loss / total}')
    draw_graph(history, name)

def draw_graph(history, name): # 3
#     plt.figure(1)
    plt.plot(history['t_loss'])
    plt.plot(history['v_loss'])
    plt.title(name + ' Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig(name+' loss graph.png')
    plt.show()

#     plt.figure(2)
    plt.plot(history['t_accu'])
    plt.plot(history['v_accu'])
    plt.title(name + ' Accuray')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig(name+' accuracy graph.png')
    plt.show()

def draw_emojis(model, w2vm, phrases, dim): # 2
    x = embedding(w2vm, phrases, dim)
    pred = model.predict(x)
    print_predictions(phrases, pred)
