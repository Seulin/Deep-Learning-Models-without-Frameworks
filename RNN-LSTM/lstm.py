import numpy as np
from module import *

class Cell_lstm(baseLayer):
        # input: input, prev_h
        # output: output, cur_h
        def sigmoid(self, x):
            return 1 / (1 +np.exp(-x))

        def sigmoid_back(self, x):
            return (1 - x) * x

        def forget_gate(self, input, params):
            self.f = self.sigmoid(np.matmul(input, params['Wf'].T) + params['Bf'])
            return self.f

        def input_gate(self, input, prev_c, f, params):
            self.i = self.sigmoid(np.matmul(input, params['Wi'].T) + params['Bi']) #b ,h
            self.g = np.tanh(np.matmul(input, params['Wg'].T) + params['Bg']) #cur_c_tilda #b, h
            return np.multiply(f, prev_c) + np.multiply(self.i, self.g) #cur_c # b, h

        def output_gate(self, input, cur_c, params):
            self.o = self.sigmoid(np.matmul(input, params['Wo'].T) + params['Bo'])
            self.tanh_c = np.tanh(cur_c)
            return np.multiply(self.o, self.tanh_c) #cur_h # b, h

        def forward(self, input, prev_h, prev_c, params):
#             assert input.shape == (RNN.size['b'], RNN.size['i'] or ['h'])
            assert prev_h.shape == (LSTM.size['b'], LSTM.size['h'])
            assert prev_c.shape == (LSTM.size['b'], LSTM.size['h'])

            concat = np.concatenate((input, prev_h), axis = 1) # b, i+h
            f = self.forget_gate(concat, params) # b, h
            cur_c = self.input_gate(concat, prev_c, f, params) #b, h
            cur_h = self.output_gate(concat, cur_c, params)

            assert cur_c.shape == prev_c.shape
            assert cur_h.shape == prev_h.shape

            self.save(concat, prev_c)

            return cur_h, cur_c

        def save(self, input, prev_c):
            super().save(input)
            self.prev_c = prev_c

        def backward(self, o_grad, h_grad, c_grad, params):
            # y_grad.shape == b, o
            # h_grad.shape == b, h
            # c_grad.shape == b, h
            mul=1
            y_grad = h_grad + o_grad #final output_grad

            # output gate
            o_grad = np.multiply(y_grad, self.tanh_c) # b, h
            o_grad = self.sigmoid_back(self.o)*o_grad #b, h
            Wo_grad = np.matmul(o_grad.T, self.input)*mul # h, i+h
            Bo_grad = np.mean(o_grad, axis = 0) # h
            concat_grad = np.matmul(o_grad, params['Wo']) # b, i+h

            # input gate
            tanh_c_grad = np.multiply(y_grad, self.o) # b,h
            cur_c_grad = np.multiply(1 - np.tanh(self.tanh_c)**2, self.tanh_c) + c_grad #b, h
            i_grad = np.multiply(cur_c_grad, self.g) #b, h
            g_grad = np.multiply(cur_c_grad, self.i) #b, h

            i_grad = self.sigmoid_back(self.i)*i_grad
            g_grad = (1 - np.tanh(self.g)**2)*g_grad
            Wi_grad = np.matmul(i_grad.T, self.input)*mul# b, i+h
            Wg_grad = np.matmul(g_grad.T, self.input)*mul# b, i+h
            Bi_grad = np.mean(i_grad, axis = 0) # h
            Bg_grad = np.mean(g_grad, axis = 0) #h
            concat_grad += np.matmul(i_grad, params['Wi'])
            concat_grad += np.matmul(g_grad, params['Wg'])

            # forget gate
            prev_c_grad = np.multiply(cur_c_grad, self.f) #b, h
            f_grad = np.multiply(cur_c_grad, self.prev_c) #b,h
            f_grad = self.sigmoid_back(self.f) * f_grad
            Wf_grad = np.matmul(f_grad.T, self.input)*mul# b, i+h
            Bf_grad = np.mean(f_grad, axis = 0) # h
            concat_grad += np.matmul(f_grad, params['Wf'])

            assert concat_grad.shape in [(LSTM.size['b'], LSTM.size['i']+LSTM.size['h']),
                                          (LSTM.size['b'], LSTM.size['h']+LSTM.size['h'])]
            return Wf_grad, Wi_grad, Wg_grad, Wo_grad, Bf_grad, Bi_grad, Bg_grad, Bo_grad, \
                    concat_grad[:, :-LSTM.size['h']], concat_grad[:, -LSTM.size['h']:], \
                    prev_c_grad


class Column_lstm:
        # input: input, prev_hs
        # output: output, cur_hs
        def __init__(self, l_size):
            self.cells = [] # k-th cells
            for i in range(l_size):
                self.cells.append(Cell_lstm())

        def forward(self, input, prev_hs, prev_cs, params):
            assert input.shape == (LSTM.size['b'], LSTM.size['i'])
            assert prev_hs.shape == (LSTM.size['l'], LSTM.size['b'], LSTM.size['h'])
            assert prev_cs.shape == (LSTM.size['l'], LSTM.size['b'], LSTM.size['h'])
            data = input

            cur_hs = np.zeros(prev_hs.shape)
            cur_cs = np.zeros(cur_hs.shape)
            for i, cell in enumerate(self.cells):
                data, cur_c = cell.forward(data, prev_hs[i], prev_cs[i], params[i])
                cur_hs[i] = data
                cur_cs[i] = cur_c

            return data, cur_hs, cur_cs

        def backward(self, o_grad, hs_grad, cs_grad, params):
            # o_grad.shape == b, o
            # hs_grad.shape == l, b, h

            grad_sets = []
            prev_hs_grad = np.zeros(hs_grad.shape)
            prev_cs_grad = np.zeros(cs_grad.shape)
            for i, cell in enumerate(reversed(self.cells)):
                grad_set = cell.backward(o_grad, hs_grad[i], cs_grad[i], params[-(i+1)])
                o_grad = np.zeros(o_grad.shape) #grad_set[-3] # i_grad
                prev_hs_grad[i] = grad_set[-2] # prev_h_grad
                prev_cs_grad[i] = grad_set[-1] #prev_c_grad
                grad_sets.insert(0, grad_set[:8])

            return grad_sets, prev_hs_grad, prev_cs_grad

class LSTM(paramLayer):
    size = {}

    def __init__(self, i_size, h_size, l_size, optim, learning_rate = 0.001, momentum = 0.9):
        self.params = [] # list of dict
        self.velocity = []
        self.columns = []
        self.optim = optim
        self.momentum = momentum
        self.learning_rate = learning_rate

        LSTM.size = {'i': i_size, 'h':h_size, 'l':l_size}

        self.initialize(i_size, h_size, l_size, optim)

    def initialize(self, i_size, h_size, l_size, optim):
        o_size = h_size # to specify which one is input
        for i in range(l_size):
            params = {}
            mul = 0.2
            if i == 0:
                i = h_size + i_size
                params['Wf'] = np.random.randn(h_size, i)*mul # out, in
                params['Wi'] = np.random.randn(h_size, i)*mul # out, in
                params['Wg'] = np.random.randn(h_size, i)*mul # out, in
                params['Wo'] = np.random.randn(h_size, i)*mul # out, in
            else:
                i = h_size*2
                params['Wf'] = np.random.randn(o_size, i)*mul # out, in
                params['Wi'] = np.random.randn(o_size, i)*mul # out, in
                params['Wg'] = np.random.randn(o_size, i)*mul # out, in
                params['Wo'] = np.random.randn(o_size, i)*mul # out, in

            params['Bf'] = np.zeros(o_size) # out
            params['Bi'] = np.zeros(o_size) # out
            params['Bg'] = np.zeros(o_size) # out
            params['Bo'] = np.zeros(o_size) # out

            self.params.append(params)

            if optim == 'adam':
                velocity = {}
                for key in params:
                    velocity[key] = np.zeros(params[key].shape)
                self.velocity.append(velocity)

    def forward(self, inputs):
#         assert inputs.shape[0] == 10 #길이
        assert inputs.shape[2] == LSTM.size['i'] #인풋 피처
        LSTM.size['b'] = inputs.shape[1] # last input may differ
        LSTM.size['w'] = inputs.shape[0] # 단어 개수 =10 # always same

        if not self.columns: # empty
            for i in range(LSTM.size['w']):
                self.columns.append(Column_lstm(LSTM.size['l']))

        # initialize
        prev_hs = np.zeros((LSTM.size['l'], LSTM.size['b'], LSTM.size['h']))
        prev_cs = np.zeros((LSTM.size['l'], LSTM.size['b'], LSTM.size['h']))
        for i, column in enumerate(self.columns):
            data, prev_hs, prev_cs = column.forward(inputs[i], prev_hs, prev_cs, self.params)

        return data # last output

    def backward(self, grad):
        def add_grad_sets(origin, new):
            for i, layer in enumerate(new): #layer
                for j, grad in enumerate(layer): # each
                    origin[i][j] += grad # new[i][j]
            return origin

        lst_grad_sets = []
        zero_grad = np.ones(grad.shape)
        hs_grad = np.zeros((LSTM.size['l'], LSTM.size['b'], LSTM.size['h']))
        cs_grad = np.zeros((LSTM.size['l'], LSTM.size['b'], LSTM.size['h']))
        for i, column in enumerate(reversed(self.columns)):
            grad_sets, hs_grad, cs_grad = column.backward(grad, hs_grad, cs_grad, self.params)
            grad = zero_grad
            lst_grad_sets.append(grad_sets)

        self.update(lst_grad_sets)

    def update(self, lst_grad_sets):
        for grad_sets in lst_grad_sets:
            for l in range(len(grad_sets)): #layer
                # Wih_grad, Whh_grad, Who_grad, Bih_grad, Bhh_grad, Bho_grad = grad_set
                for i, name in enumerate(['Wf', 'Wi', 'Wg', 'Wo', 'Bf', 'Bi', 'Bg', 'Bo']):
                    if self.optim == 'sgd':
                        self.params[l][name] -= self.learning_rate * grad_sets[l][i]
                    elif self.optim == 'adam':
                        self.velocity[l][name] = self.momentum* self.velocity[l][name] + \
                                                 self.learning_rate * grad_sets[l][i]
                        self.params[l][name] -= self.velocity[l][name]
