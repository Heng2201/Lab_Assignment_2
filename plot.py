import matplotlib.pyplot as plt
import numpy as np
import os

def plot_curve(train_acc, test_acc, loss, epoch, pre_epoch, learning_rate, momentum, batch_size):
    t = np.arange(0, epoch + pre_epoch)
    
    fig = plt.figure(figsize = (10,6), dpi = 100)
    fig.suptitle("Accuracy and Loss using structure B with lr = {}, momentum = {}, batch size = {}".format(learning_rate, momentum,batch_size))

    ax1 = fig.add_subplot((121))
    ax2 = fig.add_subplot((122))

    ax1.plot(t, train_acc, color = "blue", label = "Training accuracy")
    ax1.plot(t, test_acc, color = "red", label = "Testing accuracy")
    ax2.plot(t, loss, label = "Training loss")

    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch", fontsize=14)
    ax1.set_ylabel("accuracy", fontsize=14)
    ax1.annotate('train_acc:({}, {:.2f}%)'.format(t[-1]+1, float(train_acc[-1])) , xy = (t[-1] , float(train_acc[-1] )),\
                  xytext = (t[-1] , float(train_acc[-1])),\
                  horizontalalignment = "right", verticalalignment = "bottom") 
    ax1.annotate('test_acc:({}, {:.2f}%)'.format(t[-1]+1, float(test_acc[-1])) , xy = (t[-1] , float(test_acc[-1] )),\
                  xytext = (t[-1] , float(test_acc[-1])),\
                  horizontalalignment = "right", verticalalignment = "top") 
    ax1.legend()

    ax2.set_title("Loss")
    ax2.set_xlabel("epoch", fontsize=14)
    ax2.set_ylabel("Loss", fontsize=14)
    ax2.annotate('loss:({}, {:.2f})'.format(t[-1]+1, float(loss[-1])) , xy = (t[-1] , float(loss[-1] )),\
                  xytext = (t[-1] , float(loss[-1])),\
                  horizontalalignment = "right", verticalalignment = "top") 
    ax2.legend()
    fig.tight_layout()

    try:
        plt.savefig('./fig/structure_B_aug/dropout_sche_allCNN/lr{}/momentum{}_b{}/e{}.png'.format(learning_rate, momentum, batch_size, epoch + pre_epoch))
    
    except FileNotFoundError:
        os.makedirs('./fig/structure_B_aug/dropout_sche_allCNN/lr{}/momentum{}_b{}/'.format(learning_rate, momentum, batch_size))
        plt.savefig('./fig/structure_B_aug/dropout_sche_allCNN/lr{}/momentum{}_b{}/e{}.png'.format(learning_rate, momentum, batch_size, epoch + pre_epoch))
    plt.close()
    pass

def imgshow(img, labels, predicted, classes):
    fig = plt.figure(figsize = (10,6), dpi = 100)
    fig.suptitle("Result of random 10 images prediction",fontsize = 16)
    for i, _ in enumerate(img):
        img[i][0] = (img[i][0] * 0.247) + 0.491
        img[i][1] = (img[i][1] * 0.243) + 0.482
        img[i][2] = (img[i][2] * 0.261) + 0.447
    npimg = img.numpy()

    for i, image in enumerate(npimg):
        fig.add_subplot(2, 5, i + 1, title ="p:{} a:{}".format(classes[predicted[i]], classes[labels[i]]))

        plt.imshow(np.transpose(image,(1, 2, 0)))
    plt.show()
    
    