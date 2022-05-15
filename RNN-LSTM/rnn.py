import numpy as np
from module import *

class Cell_rnn(baseLayer):
        # input: input, prev_h
        # output: output, cur_h

        def forward(self, input, prev_h, params):
#             assert input.shape == (RNN.size['b'], RNN.size['i'])
            assert prev_h.shape == (RNN.size['b'], RNN.size['h'])

            temp_h = np.matmul(input, params['Wih'].T) + np.matmul(prev_h, params['Whh'].T)
            raw_h = temp_h + params['Bih'] + params['Bhh'] # save for back propo
            cur_h = np.tanh(raw_h)
            assert cur_h.shape == prev_h.shape

            output = np.matmul(cur_h, params['Who'].T) + params['Bho']
            assert output.shape == (RNN.size['b'], RNN.size['h'])
            # shape = b_size, o_size

            self.save(input, prev_h, raw_h, cur_h)

            return output, cur_h

        def save(self, input, prev_h, raw_h, cur_h):
            super().save(input)
            self.prev_h = prev_h
            self.raw_h = raw_h
            self.cur_h = cur_h

        def backward(self, o_grad, h_grad, params):
            # o_grad.shape == b, o
            # h_grad.shape == b, h
            a = 1
            Who_grad = np.matmul(o_grad.T, self.cur_h)/a #o, h
            Bho_grad = np.mean(o_grad, axis = 0) #o

            assert Bho_grad.shape == (RNN.size['h'],)

            cur_h_grad = np.matmul(o_grad, params['Who']) + h_grad #b, h
            raw_h_grad = np.multiply(1 - np.tanh(self.raw_h)**2, cur_h_grad) #b, h

            Bih_grad = np.mean(raw_h_grad, axis = 0) #h
            Bhh_grad = Bih_grad #h

            assert Bih_grad.shape == (RNN.size['h'],)

            Wih_grad = np.matmul(raw_h_grad.T, self.input)/a #h, i or o, h
            i_grad = np.matmul(raw_h_grad, params['Wih']) #b, i

            Whh_grad = np.matmul(raw_h_grad.T, self.prev_h)/a #h, h
            prev_h_grad = np.matmul(raw_h_grad, params['Whh']) #b, h

            return Wih_grad, Whh_grad, Who_grad, Bih_grad, Bhh_grad, Bho_grad, \
                    i_grad, prev_h_grad


class Column_rnn:
        # input: input, prev_hs
        # output: output, cur_hs
        def __init__(self, l_size):
            self.cells = [] # k-th cells
            for i in range(l_size):
                self.cells.append(Cell_rnn())

        def forward(self, input, prev_hs, params):
            assert input.shape == (RNN.size['b'], RNN.size['i'])
            assert prev_hs.shape == (RNN.size['l'], RNN.size['b'], RNN.size['h'])
            data = input

            cur_hs = np.zeros(prev_hs.shape)
            for i, cell in enumerate(self.cells):
                data, cur_h = cell.forward(data, prev_hs[i], params[i])
                cur_hs[i] = cur_h

            return data, cur_hs

        def backward(self, o_grad, hs_grad, params):
            # o_grad.shape == b, o
            # hs_grad.shape == l, b, h

            grad_sets = []
            prev_hs_grad = np.zeros(hs_grad.shape)
            for i, cell in enumerate(reversed(self.cells)):
                grad_set = cell.backward(o_grad, hs_grad[i], params[-(i+1)])
                o_grad = grad_set[-2] # i_grad
                prev_hs_grad[i] = grad_set[-1] # prev_h_grad
                grad_sets.insert(0, grad_set[:6])

            return grad_sets, prev_hs_grad


class RNN(paramLayer):
    size = {}

    def __init__(self, i_size, h_size, l_size, learning_rate = 0.001):
        self.params = [] # list of dict
        self.columns = []
        self.learning_rate = learning_rate

        RNN.size = {'i': i_size, 'h':h_size, 'l':l_size}

        self.initialize(i_size, h_size, l_size)

    def initialize(self, i_size, h_size, l_size):
        o_size = h_size # to specify which one is input
        for i in range(l_size):
            params = {}
            params['Whh'] = np.random.randn(h_size, h_size)*0.08 # out, in

            if i == 0:
                params['Wih'] = np.random.randn(h_size, i_size)*0.08 # out, in
            else:
                params['Wih'] = np.random.randn(h_size, o_size)*0.08 # out, in

            params['Who'] = np.random.randn(o_size, h_size)*0.08 # out, in
            params['Bih'] = np.zeros(h_size) # out
            params['Bhh'] = np.zeros(h_size) # out
            params['Bho'] = np.zeros(o_size) # out
            self.params.append(params)

    def forward(self, inputs):
#         assert inputs.shape[0] == 10 #길이
        assert inputs.shape[2] == RNN.size['i'] #인풋 피처
        RNN.size['b'] = inputs.shape[1] # last input may differ
        RNN.size['w'] = inputs.shape[0] # 단어 개수 =10 # always same

        if not self.columns: # empty
            for i in range(RNN.size['w']):
                self.columns.append(Column_rnn(RNN.size['l']))

        # initialize
        prev_hs = np.zeros((RNN.size['l'], RNN.size['b'], RNN.size['h']))
        for i, column in enumerate(self.columns):
            data, prev_hs = column.forward(inputs[i], prev_hs, self.params)

        return data#, prev_hs # last output

    def backward(self, grad):
        def add_grad_sets(origin, new):
            for i, layer in enumerate(new): #layer
                for j, grad in enumerate(layer): # each
                    origin[i][j] += grad # new[i][j]
            return origin

        lst_grad_sets = []
        zero_grad = np.zeros(grad.shape)
        hs_grad = np.zeros((RNN.size['l'], RNN.size['b'], RNN.size['h']))
        for i, column in enumerate(reversed(self.columns)):
            grad_sets, hs_grad = column.backward(grad, hs_grad, self.params)
            grad = zero_grad
            lst_grad_sets.append(grad_sets)

        self.update(lst_grad_sets)

    def update(self, lst_grad_sets):
        for grad_sets in lst_grad_sets:
            for l in range(len(grad_sets)): #layer
                # Wih_grad, Whh_grad, Who_grad, Bih_grad, Bhh_grad, Bho_grad = grad_set
                for i, name in enumerate(['Wih', 'Whh', 'Who', 'Bih', 'Bhh', 'Bho']): # Bhh
                    self.params[l][name] -= self.learning_rate * grad_sets[l][i]
