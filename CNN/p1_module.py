import numpy as np
from one_hot import one_hot_decoder

LR_linear = 0.001
LR_conv = 0.0001

class baseLayer():
    def __init__(self):
        raise NotImplementedError()
        
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input): # call save()
        raise NotImplementedError()
    
    def backward(self, output_grad):
        raise NotImplementedError()
        
    def save(self, input):
        self.input = input


class paramLayer(baseLayer):
    def initialize(self):
        raise NotImplementedError()
        
    def update(self): # called by backward()
        raise NotImplementedError()


class Convolution(paramLayer):
    def __init__(self, f_size, f_num, c_size, learning_rate = LR_conv):
        self.f_size = f_size
        self.f_num = f_num
        self.c_size = c_size
        self.learning_rate = learning_rate
        
        self.initialize(f_size, f_num, c_size)
        
    def initialize(self, f_size, f_num, c_size):
        #random initialization
        self.weight = np.random.randn(f_num, c_size, f_size, f_size)
        self.bias = np.random.randn(f_num)
        
    def forward(self, input):
        shape = input.shape
        if len(shape) == 3:
            assert (self.c_size == 1)
            input = input.reshape(shape[0], 1, shape[1], shape[2])
        else:
            assert (self.c_size == input.shape[1]) 
        assert(len(input.shape) == 4)

        res = self.convolution(input, self.weight) # (b_size, f_num, o_size, o_size)
        
        res = self.add_bias(res, self.bias)
        self.save(input)
        return res

    def add_bias(self, target, bias):
        assert(target.shape[1] == self.f_num)
        
        b_size, c_size, _, _ = target.shape
        for b in range(b_size):
            for c in range(c_size):
                target[b,c] += bias[c]
        return target
    
    
    def convolution(self, input, filter, auto_pad = False): # do convolution without bias
        b_size, c_size, i_size, _ = input.shape
        f_num, c_size2, f_size, _= filter.shape
#         print(c_size, c_size2)
        assert (c_size == c_size2)
        
        pad = f_size - 1 if auto_pad else 0
        o_size = i_size - f_size + 2*pad + 1 # stride = 1

        i_extended = self.im2col(input, f_size, o_size, pad) # input_extended        
        f_extended = self.fi2row(filter) # filter_extended

        res = np.matmul(f_extended, i_extended)

        # coloumn to image
        assert(res.shape[0:2] == (b_size, f_num))
        res = res.reshape(b_size, f_num, o_size, o_size)
        return res
            

    def im2col(self, img, f_size, o_size, pad = 0): #input.shape = (n, c, i, i)
        b_size, c_size, i_size, _ = img.shape
        f_total = f_size*f_size

        img = np.pad(img, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant', constant_values = 0)
        col = np.zeros((b_size, c_size*f_total, o_size*o_size))
        
        for b in range(b_size):
            for c in range(c_size):
                for y in range(o_size):
                    for x in range(o_size):
                        col[b][c*f_total:c*f_total+f_total, x+o_size*y] = img[b, c][y:y+f_size, x:x+f_size].reshape(f_total)
        return col

    def fi2row(self, weight): # channel size of input
        f_num, c_size, f_size, _ = weight.shape
        f_total = f_size*f_size
        weight = weight.reshape(f_num, f_total*c_size)
        
        return weight

    
    def backward(self, grad):
        i_grad = self.convolution(grad, self.weight.transpose(1,0,3,2), auto_pad = True)
        
        re_input = self.reshape01(self.input)
        re_grad = self.reshape01(grad)
        w_grad = self.convolution(re_input, re_grad)
        w_grad = self.reshape01(w_grad)
        
        b_grad = np.sum(grad, axis = (0, 2, 3))
        assert (self.input.shape == i_grad.shape)
        assert (self.weight.shape == w_grad.shape)
        assert (self.bias.shape == b_grad.shape)
        
        self.update(w_grad, b_grad)
        return i_grad
    
    def update(self, w_grad, b_grad):
        self.weight -= self.learning_rate * w_grad
        self.bias -= self.learning_rate * b_grad
    
    def reshape01(self, mat):
        shape = mat.shape
        return mat.transpose(1,0,2,3)
    

class Linear(paramLayer):
    def __init__(self, in_features, out_features, learning_rate = LR_linear):
        self.in_f = in_features
        self.out_f = out_features
        self.learning_rate = learning_rate
        
        self.initialize()
        
    def initialize(self):
        #random initialization
        self.weight = np.random.randn(self.out_f, self.in_f)
        self.bias = np.zeros((1, self.out_f))
        
    def forward(self, input):
        shape = input.shape
        if len(shape) == 4: # after maxpool layer
            input = input.reshape(shape[0], shape[1]*shape[2]*shape[3])
        elif len(shape) == 3: # common linear input
            input = input.reshape(shape[0], shape[1]*shape[2])
        assert(len(input.shape) == 2)
        assert(input.shape[1] == self.in_f)

        self.save(input)
        
        without_bias = np.dot(self.input, self.weight.T)
        return np.add(without_bias, self.bias)
    
    def backward(self, grad):
        i_grad = np.dot(grad, self.weight) # input gradient
        w_grad = np.dot(grad.T, self.input)
        b_grad = np.sum(grad, axis = 0, keepdims = True)
        
        assert(i_grad.shape == self.input.shape)
        assert(w_grad.shape == self.weight.shape)
        assert(b_grad.shape == self.bias.shape)
        
        self.update(w_grad, b_grad)
        
        return i_grad
    
    def update(self, w_grad, b_grad):
        self.weight -= self.learning_rate * w_grad
        self.bias -= self.learning_rate * b_grad



class Maxpool(baseLayer):
    def __init__(self, p_size = 2):
        self.p_size = p_size
    
    def forward(self, input):
        b_size, c_size, i_size, _ = input.shape
        p_size = self.p_size # pooling size
        assert (i_size % p_size == 0)
        o_size = i_size // p_size
        
        res = np.zeros((b_size, c_size, o_size, o_size))
        m_grad = np.zeros((b_size, c_size, i_size, i_size)) # local gradient
        for b in range(b_size):
            for c in range(c_size):
                for y in range(o_size):
                    for x in range(o_size):
                        part = input[b, c, y*p_size:y*p_size+p_size, x*p_size:x*p_size+p_size]
                        res[b, c][y, x] = np.max(part)
                        
                        # calculate local gradient
                        index = np.argmax(part)
                        m_grad[b,c,y*p_size:y*p_size+p_size, x*p_size:x*p_size+p_size][index//p_size, index%p_size] = 1
                        
        assert (res.shape == (b_size, c_size, o_size, o_size))
        self.save(m_grad, res)

        return res
    
    def backward(self, grad):
        shape = grad.shape
        if len(shape) == 2: # after linear backward layer
            grad = grad.reshape(self.output.shape)
        assert(grad.shape == self.output.shape)
        
        b_size, c_size, o_size, _ = grad.shape
        i_size = o_size * self.p_size
        p_size = self.p_size
        g_extended = np.zeros((b_size, c_size, i_size, i_size)) # gradient extended
        for b in range(b_size):
            for c in range(c_size):
                for y in range(o_size):
                    for x in range(o_size):

                        g_extended[b, c][y*p_size:y*p_size+p_size, x*p_size:x*p_size+p_size] = grad[b,c][y,x]  

        
        res= np.multiply(self.m_grad, g_extended)
        return res
    
    def save(self, m_grad, output):
        self.m_grad = m_grad
        self.output = output

class Relu(baseLayer):
    def __init__(self):
        pass
    
    def forward(self, input):
        self.save(input)
        return np.maximum(0, self.input)
        
    def backward(self, grad):
        r_grad =  np.where(self.input > 0, 1, 0)
        return np.multiply(r_grad, grad)


class Softmax(baseLayer):
    def __init__(self):
        pass
    
    def forward(self, input):
        output = np.zeros(input.shape)
        for i, line in enumerate(input):
            new_line = line / input.shape[1] # prevent exp overflow
            exps = np.exp(new_line)
            output[i] = exps / np.sum(exps)
        
        self.save(input, output)
        return output
    
    def save(self, input, output):
        super().save(input)
        self.output = output
        
    def backward(self, grad):
        s_grad = np.multiply(self.output, (1 - self.output))
        return np.multiply(s_grad, grad)


class CELoss(baseLayer):
    def __init__(self):
        self.label = 0
        self.prob = 0
        self.loss = 0
        self.dLoss = 0
        
    def __call__(self, prob, label):
        return self.forward(prob, label)
        
    def set_data(self, prob):
        # substitute for log calculation
        prob[prob == 0] = 1e-200
        prob[prob == 1] = 9.9e-1
        return prob

    def save(self, prob, label):
        self.prob = prob
        self.label = label
        
    # caculate loss(= cost)
    def forward(self, prob, label):
        assert (label.shape == prob.shape)
        assert (label.shape[1] == 10)
        prob = self.set_data(prob)
        
        result = np.zeros((prob.shape[0], 1))
        for i, (p, l) in enumerate(zip(prob, label)):
            result[i] = - np.sum(l*np.log(p) + (1 - l)*np.log(1 - p))
        self.loss = result
        
        self.save(prob, label)
        return self.loss

    def backward(self):
        self.dLoss = -np.divide(self.label, self.prob) + np.divide(1-self.label, 1-self.prob)
        assert(self.dLoss.shape == self.prob.shape)
        return self.dLoss
    
class Avgpool(baseLayer): # GAP for extra
    def __init__(self):
        pass

class Model(baseLayer):
    def __init__(self, num_layer = 2, lr_conv = LR_conv, lr_linear = LR_linear, GAP=False):        

        r1 = Relu()
        r2 = Relu()
        m1 = Maxpool()
        m2 = Maxpool()
        
        if num_layer == 2:
            c1 = Convolution(f_size = 3, f_num = 2, c_size = 1, learning_rate = lr_conv)
            c2 = Convolution(f_size = 2, f_num = 5, c_size = 2, learning_rate = lr_conv)
            l = Linear(5*6*6, 10, lr_linear)
            self.layers = [c1, r1, m1, c2, r2, m2, l]
            
        elif num_layer == 3:
            c1 = Convolution(f_size = 3, f_num = 2, c_size = 1, learning_rate = lr_conv)
            c2 = Convolution(f_size = 2, f_num = 5, c_size = 2, learning_rate = lr_conv) # c_size = previous f_num
            c3 = Convolution(f_size = 3, f_num = 12, c_size = 5, learning_rate = lr_conv)
            r3 = Relu()
            m3 = Maxpool()
            l = Linear(12*2*2, 10, lr_linear)
            self.layers = [c1, r1, m1, c2, r2, m2, c3, r3, m3, l]
            
        s = Softmax()
        self.layers.append(s)
    
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
            
    def save(self, path):
        params = [] # [weight_1, bias_1, weight_2, bias_2, ...]
        for layer in self.layers:
            if isinstance(layer, paramLayer):
                params.append(layer.weight)
                params.append(layer.bias)
        np.save(path, params)
    
    def load(self, path):
        params = np.load(path, allow_pickle = True)
        print(params)
        index = 0
        for layer in self.layers:
            if isinstance(layer, paramLayer):
                layer.weight = params[index]
                layer.bias = params[index + 1]
                index += 2

                