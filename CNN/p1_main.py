from p1_data import import_img, import_lbl, import_img, import_lbl, \
                data_loader
import numpy as np
from one_hot import one_hot_encoder, one_hot_decoder

from time import time
from p1_module \
    import Convolution, Linear, Maxpool, Relu, Softmax, CELoss, Model

from torch.utils.tensorboard import SummaryWriter
from analyze import draw_cm, tensorboard_top3


train_img = import_img(3000, './mnist_data/train-images-idx3-ubyte.gz')
train_lbl = import_lbl(3000, './mnist_data/train-labels-idx1-ubyte.gz')
test_img = import_img(200, './mnist_data/t10k-images-idx3-ubyte.gz')
test_lbl = import_lbl(200, './mnist_data/t10k-labels-idx1-ubyte.gz')


# Normalization (0~255) to (0~1)
train_img, test_img = train_img / 255.0, test_img / 255.0

# One hot encoding
train_lbl = one_hot_encoder(train_lbl)
test_lbl = one_hot_encoder(test_lbl)

print("trainging image :",  train_img.shape)
print("trainging label :",  train_lbl.shape)
print("testing image : ", test_img.shape)
print("testing label : ", test_lbl.shape)

# grobal variable
writer = SummaryWriter('runs/implementation')
epochs = 5
batch_size = 128
# default conv lr = 0.0001, linear lr = 0.001


def train(model, loss_func, name, batch_size = batch_size):
    # include saving loss graph
    batches = data_loader(train_img, train_lbl, batch_size)
    batches = list(batches)
    batch_len = len(batches)
    
    print('--Loss & Running time--')
    for epoch in range(epochs):
        total_loss = 0
        start_time = time()
        for i, (img, lbl) in enumerate(batches):
            output = model(img)
            loss = loss_func(output, lbl)
            loss_grad = loss_func.backward()
            model.backward(loss_grad)
            
            loss = np.mean(loss)
            
            total_loss += loss
            writer.add_scalar(name+ ' training_loss', loss, epoch * batch_len + i)
            

            pred = model(test_img[:30])
            loss = loss_func(pred, test_lbl[:30])
            loss = np.mean(loss)
            writer.add_scalar(name +' testing_loss', loss, epoch * batch_len + i)
            
        end_time = time()
        elapsed = end_time - start_time 
        print(f'{epoch + 1} epoch: {total_loss / batch_len} // {elapsed}sec')
        
def test(model):
    accuracy = model.score(test_img, test_lbl)
    print('--Accuracy--')
    print(accuracy)
    return accuracy

# names for printing
names = ['CNN2', 'CNN3', 'NN2']

# CNN2
model2 = Model(lr_conv = 0.0005, lr_linear = 0.01)
loss2 = CELoss()
train(model2, loss2, names[0], 128)
test(model2)

# CNN3
model3 = Model(3, lr_conv = 0.0001, lr_linear = 0.001)
loss3 = CELoss()
train(model3, loss3, names[1], 64)
test(model3)


# NN2
model_nn = Model()
l1 = Linear(784, 182, 0.001)
l2 = Linear(182, 42, 0.001)
l3 = Linear(42, 10, 0.001)
r1 = Relu()
r2 = Relu()
s = Softmax()
c= CELoss()
model_nn.layers= [l1, r1, l2, r2, l3, s]
train(model_nn, c, names[2], batch_size = 32)
test(model_nn)


# for analysis
models = [model2, model3, model_nn]

# Plot confusion matrix for test
for model, name in zip(models, names):
    true = one_hot_decoder(test_lbl)
    pred = model.predict(test_img)
    draw_cm(true, pred, name)
    
# Top 3 scored images for test
for model, name in zip(models, names):
    prob = model(test_img)
    top3 = tensorboard_top3(writer, test_img, prob, name)
    print(top3)
