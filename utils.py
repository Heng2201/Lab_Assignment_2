import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


import os
import numpy as  np
from tqdm import tqdm 

import plot

def load_CIFAR10(dir, batch_size):
    transform = transforms.Compose([transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), transforms.ToTensor(), transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.261))])
    transform__test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.261))])

    trainset = torchvision.datasets.CIFAR10(root = dir, train = True, download = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 0)

    testset = torchvision.datasets.CIFAR10(root = dir, train = False, download = True, transform = transform__test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 0)

    classes = ("plane", 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

def set_device_to_cuda():
    if torch.cuda.is_available(): devices = torch.device('cuda')
    else: devices = torch.device('cpu')

    return devices

def train_network(net, device, criterion, optimizer, scheduler, trainloader, testloader, train_hist, test_hist, loss_hist, EPOCH, pre_epochs, lr, momentum, batch_size, save_interval = 5):
    # declare list that store accuracy and loss
    train_acc, test_acc, train_loss  = train_hist, test_hist, loss_hist
    
    for epoch in range(EPOCH):
        loader = tqdm(trainloader) 
        running_loss = 0.0
        # enumerate(iterable , start), default is 0
        for i, data in enumerate(loader, 0):
            # get the inputs; dat is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            # zero_grad() sets the gradients of all optimized torch.Tensors to zero
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # optimizer.step() update parameters
            optimizer.step() 

            # print statistics
            running_loss += loss.item()
            loader.set_description(f"epoch{epoch + 1}, {(running_loss / (i + 1)):.4f}")
            # if i % 2000 == 1999: # print every 2000 mini-batches
            #     print("[{:d}, {:5d}] loss : {:.3f}".format(epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
        train_loss.append(running_loss / (i + 1)) 
        train_acc.append(accuracy(net, trainloader, device=device))
        test_acc.append(accuracy(net, testloader, device=device))
        if scheduler != None:
            scheduler.step()
        
        
        if ((epoch + 1) % save_interval == 0):  
            save_params('./params/structure_B_aug/dropout_sche_allCNN/lr{}/momentum{}_b{}/'.format(lr, momentum, batch_size), net, optimizer, scheduler, epoch + pre_epochs + 1)
            save_hist('./history/structure_B_aug/dropout_sche_allCNN/lr{}/momentum{}_b{}/'.format(lr, momentum, batch_size), train_hist, test_hist, loss_hist, epoch + pre_epochs + 1)
            plot.plot_curve(train_hist, test_hist, loss_hist, epoch + 1, pre_epochs, lr, momentum, batch_size)

    return train_loss, train_acc, test_acc

def accuracy(network, dataloader, device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs

    # torch.no_grad() is a context-manager that disabled gradient calculation
    with torch.inference_mode():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = network(images)
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            # .item() is to get the items in this Tensor.cuda.LongTensor
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct // total

    return accuracy

def save_params( params_PATH, network, optimizer, scheduler, epoch):
    checkpoint = {'state_dict' : network, 'optimizer' : optimizer, 'scheduler' : scheduler}
    try:
        torch.save(checkpoint, params_PATH + 'checkpoint_strucB_e{}_1.pth'.format(epoch))
    except RuntimeError:
        os.makedirs(params_PATH)
        torch.save(checkpoint, params_PATH + 'checkpoint_strucB_e{}_1.pth'.format(epoch))
        
     
def save_hist(hist_PATH, training_hist, testing_hist, loss_hist, epoch):
    try:
        np.savez(hist_PATH + "history_structB_e{}_1.npz".format(epoch), training_hist = training_hist, testing_hist = testing_hist, loss_hist = loss_hist)
    except FileNotFoundError :          
        os.makedirs(hist_PATH)
        np.savez(hist_PATH + "history_structB_e{}_1.npz".format(epoch), training_hist = training_hist, testing_hist = testing_hist, loss_hist = loss_hist)

def load_params(params_PATH, pre_epoch):
    checkpoint = (torch.load(params_PATH + 'checkpoint_strucB_e{}_1.pth'.format(pre_epoch)))
    return checkpoint
    
def load_hist(hist_PATH, pre_epoch):
    history = np.load(hist_PATH + "history_structB_e{}_1.npz".format(pre_epoch))
    return history

def plot_10(network, dataloader, classes, device, batch_size):
    correct_pred = {classname : 0 for classname in classes}
    # {'plane': 0, 'car': 0, 'bird': 0, 'cat': 0, 'deer': 0, 'dog': 0, 'frog': 0, 'horse': 0, 'ship': 0, 'truck': 0}
    total_pred = {classname: 0 for classname in classes}

    with torch.inference_mode():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = network(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            # print accuracy for each class
        # dict.items() return both keys and values
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print("Accuracy for class: {:5s} is {:.1f}%".format(classname, accuracy))

        i = 0
        data = {}
        img, label  = torch.empty((0, 3, 32, 32)).to(device), torch.empty((0)).to(device)
        
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            if np.random.random(1) > 0.1: 
                print(np.random.random(1))
                continue
            img = torch.concat((img, images), 0)
            label = torch.concat((label, labels), 0)
            # calculate outputs by running images through the network
            i = i + batch_size
            if i >= 10: break 
        img = img[0:10, :, : ,:]
        label = label[0:10].to(dtype=torch.int)
        outputs = network(img)
        _, predicted = torch.max(outputs,1)
        
        plot.imgshow(img.cpu(),label.cpu(), predicted, classes)
    
def convert_to_csv(learning_rate, momentum, batch_size, pre_epochs):
    history = load_hist('./history/structure_B_aug/dropout_sche_allCNN/lr{}/momentum{}_b{}/'\
                        .format(learning_rate, momentum, batch_size), pre_epochs)
    PATH = './csv/structure_B_aug/dropout_sche_allCNN/lr{}/momentum{}_b{}/e_{}/'.format(learning_rate, momentum, batch_size, pre_epochs)
    if os.path.exists(PATH) ==False:
        os.makedirs(PATH)
    for key, value in history.items():
        np.savetxt(PATH + "{}.csv".format(key), value)

# convert_to_csv(0.0005, 0.9, 10, 100)
