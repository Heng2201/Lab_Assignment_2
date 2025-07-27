#%%
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import utils
import obj
import plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=20, help='Batch size?')
    parser.add_argument('--EPOCHS', type=int, default=200, help='Number of epochs')
    parser.add_argument('--pre_epochs', type=int, default=0, help='Start from which epochs?')
    parser.add_argument('--plot_10', type=bool, default=False, help="plot 10 images")
    
    opt = parser.parse_args()
     # Declare variables
    EPOCH = opt.EPOCHS
    # how many sample per batch to load
    batch_size = opt.batch
    learning_rate = 0.001
    momentum = 0.9
    # wd = 0.001
    pre_epochs = opt.pre_epochs
    save = False
    SUMMARY = False

    # Call dataloader
    trainloader, testloader, classes = utils.load_CIFAR10('./data', batch_size)
    # Call network model
    net = obj.Net()

    # Train using GPU
    device = utils.set_device_to_cuda()
    net.to(device)
    if SUMMARY: 
        summary(net,(3, 28, 28))
        
    
    else:
        # Declare list to store loss and accuracy history
        loss_hist, train_hist, test_loss, test_hist = [], [] ,[], []
        
        
        # Declare optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 60, 90], gamma=0.1)
        
        if pre_epochs !=0:
            checkpoint = utils.load_params('./params/structure_B_aug/dropout_sche_allCNN/lr{}/momentum{}_b{}/'.format(learning_rate, momentum, batch_size), pre_epochs)
            net ,optimizer, scheduler = checkpoint['state_dict'], checkpoint['optimizer'], checkpoint['scheduler']

            history = utils.load_hist('./history/structure_B_aug/dropout_sche_allCNN/lr{}/momentum{}_b{}/'.format(learning_rate, momentum, batch_size), pre_epochs) 
            loss_hist, train_hist, test_hist = history["loss_hist"].tolist(), history['training_hist'].tolist(), history["testing_hist"].tolist()

            print("File is opened")
        if opt.plot_10:
            utils.plot_10(net, testloader, classes, device, batch_size)

        else:

            train_loss, train_acc, test_acc = utils.train_network(net, device, criterion, optimizer, scheduler, trainloader, testloader, train_hist, test_hist, loss_hist, EPOCH, pre_epochs,\
                                                            learning_rate, momentum, batch_size)
        
        print("done")

# %%
