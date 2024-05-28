"""
                                                    ⠀⠀⠀⠀⠀⠀⠀⢠⣤⣀⠀⠀⠀⠀⢀⣀⣤⣤⠀⠀⠀⠀⠀⠀⠀
                                                    ⠀⠀⢀⢀⠀⠀⠀⢸⡿⠛⠛⠛⠛⠛⠉⠛⢿⣿⠀⠀⠀⠀⠀⠀⠀
 _   _       _     _____         _             		⠀⠠⣿⣿⣿⣄⠀⣼⠀⠀⠉⣍⣀⣀⡍⠁⠀⢹⡀⠀⠀⠀⠀⠀⠀
| \ | | ___ | |_  |_   _|__   __| | __ _ _   _ 		⠀⢸⣿⣿⣿⣿⡷⠋⠈⠀⠀⠀⠀⠀⠀⠀⠈⠘⠣⡀⠀⠀⠀⠀⠀
|  \| |/ _ \| __|   | |/ _ \ / _` |/ _` | | | |		⠀⠈⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⣦⡀⠀⠀
| |\  | (_) | |_    | | (_) | (_| | (_| | |_| |		⠀⠀⢹⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿⣦⠀
|_| \_|\___/ \__|   |_|\___/ \__,_|\__,_|\__, |		⠀⠀⣸⣿⣿⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇
                                         |___/ 		⠀⣤⡟⠛⠋⠉⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⠈⠋⠈⢿⣿⡿
                                                    ⢀⡉⠀⠀⣀⣤⣄⢈⣿⣿⣿⣿⣿⣿⣿⣿⣿⢀⣤⣤⣄⠀⠀⣴⡄
                                                    ⠘⢇⠀⠰⣿⣿⢟⢼⣿⣿⣿⣿⣿⣿⣿⣿⡿⢜⠿⠿⠿⠀⡀⠀⠀
@Author: Junjiao Sun                               
@Time : Created in 4:54 PM 2024/02/19   
@FileName: LASO_F1.py                           
@Software: PyCharm


Introduction of this File:
In this training process, we use LASO for comparison.
When we get all the 2D feature maps, we could train those data using CNN networks.
Here are some main stream CNN models include VGG, Resnet and so on.
"""

import sys
import os
sys.path.append("/home/junjiao/PycharmProjects/Empatia_Git/WEMAC")
from sklearn.metrics import f1_score

from WEMAC.Data_extraction import data_extraction
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import os
import argparse
from models import *
from WEMAC.Create_feature_maps import get_normalization, generate_feature_maps
from utils import progress_bar


def feature_map_trans(feature_maps):
    feature_maps_trans = np.zeros((feature_maps.shape[2], 1, feature_maps.shape[0], feature_maps.shape[1]))
    for i in range(feature_maps.shape[2]):
        feature_map = feature_maps[:, :, i]
        feature_maps_trans[i, 0, :, :] = feature_map
    return feature_maps_trans


def LASO_resplit(trainset, testset, Pvalue):
    # reunion all features
    set_all = np.hstack((trainset, testset))
    picked_index = np.where(set_all[124, :] == Pvalue)
    test_index = picked_index[0][:int(len(picked_index[0]) / 2)]
    test_data = set_all[:, test_index]
    train_data = np.hstack((set_all[:, :test_index[0] - 1], set_all[:, test_index[-1] + 1:]))
    return train_data, test_data


best_acc = 0  # best test accuracy
best_f1 = 0.0


def training_process_laso(net_type, P_index, featureMap_stratgy):
    parser = argparse.ArgumentParser(description='PyTorch Empatia Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    best_f1 = 0.0

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')

    log_file = '../log_data_normalization_functions_KNN_AD.log'
    data_file = '../Pack_all_data.json'

    train_array_all, test_array_all = get_normalization(data_file, log_file)
    # Doing LASO to resplit the train and test
    train_data, test_data = LASO_resplit(trainset=train_array_all, testset=test_array_all, Pvalue=P_index)
    train_array_all = train_data.copy()
    test_array_all = test_data.copy()

    test_feature_map, test_label = generate_feature_maps(test_array_all, strategy=featureMap_stratgy)
    #
    train_feature_map, train_label = generate_feature_maps(train_array_all, strategy=featureMap_stratgy)
    if featureMap_stratgy == 'All_concat':
        test_feature_map1, test_label1 = generate_feature_maps(test_array_all, strategy='AllFromOne')
        test_feature_map2, test_label2 = generate_feature_maps(test_array_all, strategy='HalfAndHalf')
        test_feature_map3, test_label3 = generate_feature_maps(test_array_all, strategy='HalfAndRandom')
        #
        train_feature_map1, train_label1 = generate_feature_maps(train_array_all, strategy='AllFromOne')
        train_feature_map2, train_label2 = generate_feature_maps(train_array_all, strategy='HalfAndHalf')
        train_feature_map3, train_label3 = generate_feature_maps(train_array_all, strategy='HalfAndRandom')

        test_feature_map = np.dstack((test_feature_map1, test_feature_map2))
        test_feature_map = np.dstack((test_feature_map, test_feature_map3))
        test_label = np.hstack((test_label1, test_label2))
        test_label = np.hstack((test_label, test_label3))
        train_feature_map = np.dstack((train_feature_map1, train_feature_map2))
        train_feature_map = np.dstack((train_feature_map, train_feature_map3))
        train_label = np.hstack((train_label1, train_label2))
        train_label = np.hstack((train_label, train_label3))

    # transfer feature maps into the correct form (c, d, w, h)
    x_test = feature_map_trans(test_feature_map)
    x_train = feature_map_trans(train_feature_map)
    # Convert the ndarray to a PyTorch tensor
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(np.array(train_label).astype(int))

    # Create a TensorDataset object
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)

    # Create a DataLoader object
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Convert the ndarray to a PyTorch tensor
    x_test_tensor = torch.from_numpy(x_test)
    y_test_tensor = torch.from_numpy(np.array(test_label).astype(int))

    # Create a TensorDataset object
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    # Create a DataLoader object
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Model
    print('==> Building model..')

    net = net_type.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     net.load_state_dict(checkpoint['net'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        print('\nTraining')
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        print('\nTesting')
        global best_acc
        global best_f1
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        prob_all = []
        label_all = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.type(torch.cuda.FloatTensor)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                prob = outputs.cpu().numpy()
                prob_all.extend(np.argmax(prob, axis=1))
                label = targets.cpu().numpy()
                label_all.extend(label)
                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Calculate the F1 score
        f1 = f1_score(label_all, prob_all, average='weighted')
        # Save checkpoint.
        acc = 100. * correct / total
        F1 = 100. * f1
        # print('F1:' + str(F1))
        if acc > best_acc:
            # print('Saving..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            # torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        if F1 > best_f1:
            best_f1 = F1
        print('Best ACC: ------- ' + str(best_acc))
        print('Best F1: ------- ' + str(best_f1))
        return acc, F1

    total_best = 0
    total_f1 = 0
    for epoch in range(start_epoch, start_epoch + 50):
        train(epoch)
        now_acc, now_f1 = test(epoch)
        scheduler.step()
        if now_acc > total_best:
            total_best = now_acc
        if now_f1 > total_f1:
            total_f1 = now_f1
    return total_best, total_f1


if __name__ == '__main__':
    data_file = '../Pack_all_data.json'
    train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label, \
        train_V_label, test_V_label = data_extraction(data_file)
    all_P_set = set(train_P_label).union(set(test_P_label))
    all_P = list(all_P_set)
    all_P_list = []
    for P in all_P:
        all_P_list.append(int(P.replace('P', '')))
    classes = ('not_fear', 'fear')
    P_acc_list = []
    P_f1_list = []
    Model_name = ''
    # Decide the feature maps generation strategy:
    strategy_list = ['AllFromOne', 'HalfAndHalf', 'HalfAndRandom', 'All_concat']
    # strategy_list = ['AllFromOne']
    # Decide the feature maps generation strategy:
    # strategy_list = ['AllFromOne', 'HalfAndHalf', 'HalfAndRandom', 'All_concat']
    model_group = [VGG('VGG19', num_classes=len(classes)), ResNet18(num_classes=len(classes)),
                   GoogLeNet(num_classes=len(classes)), DenseNet121(num_classes=len(classes)),
                   MobileNet(num_classes=len(classes)), ResNeXt29_2x64d(num_classes=len(classes))]
    for net in model_group:
        for strategy in strategy_list:
            for P_value in all_P_list:
                # K-fold cross validation.
                K = 5
                bestAcc_list = []
                bestF1_list = []
                for cross_index in range(K):
                    print('\nStart_Model:' + str(net.__class__.__name__))
                    print('Start_P: ' + str(P_value))
                    print('Start_strategy: ' + strategy)
                    print('Start_Kfold: ' + str(cross_index + 1))
                    Model_name = str(net.__class__.__name__)
                    best_acc = 0  # best test accuracy
                    best_f1 = 0.0  # best test accuracy
                    now_best, now_f1 = training_process_laso(net, P_index=P_value, featureMap_stratgy=strategy)
                    bestAcc_list.append(now_best)
                    bestF1_list.append(now_f1)
                    print('ACC list:')
                    print(bestAcc_list)
                    print('F1 list:')
                    print(bestF1_list)
                # del now_best, now_f1, net
                # torch.cuda.empty_cache()
                print('\nModel name:' + Model_name)
                print('P name:' + str(P_value))
                print('Strategy: ' + strategy)
                print('bestAcc_list: ')
                print(bestAcc_list)
                print('Acc std:' + str(np.std(bestAcc_list)))
                print('Mean best acc:' + str(np.mean(bestAcc_list)))
                P_acc_list.append(np.mean(bestAcc_list))
                print('P acc list:')
                print(P_acc_list)
                print('bestF1_list: ')
                print(bestF1_list)
                print('F1 std:' + str(np.std(bestF1_list)))
                print('Mean best F1:' + str(np.mean(bestF1_list)))
                P_f1_list.append(np.mean(bestF1_list))
                print('P f1 list:')
                print(P_f1_list)
        print('Over all acc list:')
        print(P_acc_list)
        print('Over all f1 list:')
        print(P_f1_list)
        print('Over all acc mean:')
        print(np.mean(P_acc_list))
        print('Over all f1 mean:')
        print(np.mean(P_f1_list))
        print('Over all acc std:')
        print(np.std(P_acc_list))
        print('Over all f1 std:')
        print(np.std(P_f1_list))

