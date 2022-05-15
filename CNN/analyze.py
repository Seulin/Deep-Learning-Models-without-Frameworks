# Plot confusion matrix for test
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np


def draw_cm(true, pred, name):
            conf_mat = confusion_matrix(true, pred)
            conf_mat = conf_mat/ np.sum(conf_mat, axis = 0)

            sn.heatmap(conf_mat, annot=True, fmt='.2f', cmap='GnBu')
            plt.title(name)
            plt.xlabel('Predicted')
            plt.ylabel('True')
        #     plt.savefig(act+'_cf.png')
            plt.show()

def tensorboard_top3(writer, img, prob, name):
    # prob.shape = (data_len, 10)
    res_pos = []
    res_prob = []
    for i in range(3):
        res_pos.append(np.argmax(prob, axis = 0))
        res_prob.append(np.max(prob, axis = 0))
        enum = list(enumerate(res_pos[i]))
        for label, pos in enum:
            prob[pos, label] = 0 # remove previous top
        
    for i, (pos, probs) in enumerate(zip(res_pos, res_prob)):
        images = img[pos].reshape(10, 1, 28, 28)
        writer.add_images(f'{name} Top {i+1}', images)
        
    return res_prob
