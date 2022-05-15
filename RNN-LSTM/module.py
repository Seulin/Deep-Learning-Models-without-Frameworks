import numpy as np

class baseLayer():
    def __init__(self):
        pass

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input): 
        raise NotImplementedError()

    def backward(self, output_grad):
        raise NotImplementedError()

    def save(self, input): # called by forward()
        self.input = input

class paramLayer(baseLayer):
    def initialize(self):
        raise NotImplementedError()

    def update(self): # called by baseLayer.backward()
        raise NotImplementedError()

class Linear(paramLayer):
    def __init__(self, in_features, out_features, learning_rate = 0.001):
        self.in_f = in_features
        self.out_f = out_features
        self.learning_rate = learning_rate

        self.initialize()

    def initialize(self):
        #random initialization
        self.weight = np.random.randn(self.out_f, self.in_f)*0.1
        self.bias = np.zeros((1, self.out_f))

    def forward(self, input):
        assert(len(input.shape) == 2)
        assert(input.shape[1] == self.in_f)

        self.save(input)

        without_bias = np.dot(self.input, self.weight.T)
        return np.add(without_bias, self.bias)

    def backward(self, grad):
        i_grad = np.dot(grad, self.weight) # input gradient
        w_grad = np.dot(grad.T, self.input)
        b_grad = np.mean(grad, axis = 0, keepdims = True)

        assert(i_grad.shape == self.input.shape)
        assert(w_grad.shape == self.weight.shape)
        assert(b_grad.shape == self.bias.shape)

        self.update(w_grad, b_grad)

        return i_grad

    def update(self, w_grad, b_grad):
        self.weight -= self.learning_rate * w_grad
        self.bias -= self.learning_rate * b_grad

class Softmax(baseLayer):
    def forward(self, input):
        output = np.zeros(input.shape)
        for i, line in enumerate(input):
            new_line = line - np.max(line) # prevent exp overflow
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

class CrossEntropyLoss(baseLayer):
    def __call__(self, prob, label):
        return self.forward(prob, label)

    def save(self, prob, label):
        self.prob = prob
        self.label = label

    # caculate loss
    def forward(self, prob, label):
        assert (label.shape == prob.shape)
        assert (label.shape[1] == 5)
        
        # substitute for log calculation
        prob[prob == 0] = 1e-200
        prob[prob == 1] = 9.9e-1

        loss = np.zeros((prob.shape[0], 1))
        for i, (p, l) in enumerate(zip(prob, label)):
            loss[i] = - np.sum(l*np.log(p))# + (1 - l)*np.log(1 - p))

        self.save(prob, label)
        return loss

    def backward(self):
        loss_grad = -np.divide(self.label, self.prob) + np.divide(1-self.label, 1-self.prob)
        assert(loss_grad.shape == self.prob.shape)
        return loss_grad

class Dropout(baseLayer):
    def __init__(self):
        pass

    def forward(self, input, rate = 0.5):
        assert 0 <= rate <= 1
        self.mask = np.random.uniform(0, 1, input.shape) > rate
        return self.mask*input / (1.0-rate)

    def backward(self, grad):
        return grad * self.mask
