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
@Time : Created in 3:22 PM 2024/02/20   
@FileName: train_rfe_wesad_fisher_2C.py                           
@Software: PyCharm


Introduction of this File:
Here are the training process of WESAD 3 classes with RFE and Fisher score.
"""

'''Train CIFAR10 with PyTorch.'''

import sys
sys.path.append("/home/junjiao/PycharmProjects/Empatia/")
from WESAD.Abnormal_extraction_weasd import data_extraction_wesad
from WESAD.Create_feature_maps_wesad import generate_feature_maps_wesad, get_normalization_wesad
from random import shuffle

from sklearn.metrics import f1_score

from utils import Feature_selection_fisher_score_wesad
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from models import *

def feature_map_trans(feature_maps):
    feature_maps_trans = np.zeros((feature_maps.shape[2], 1, feature_maps.shape[0], feature_maps.shape[1]))
    for i in range(feature_maps.shape[2]):
        feature_map = feature_maps[:, :, i]
        feature_maps_trans[i, 0, :, :] = feature_map
    return feature_maps_trans


best_acc = 0  # best test accuracy
best_f1 = 0.0


def training_process_rfe(net_type, featureMap_stratgy, train_array_all, test_array_all):
    parser = argparse.ArgumentParser(description='PyTorch Empatia Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    best_f1 = 0.0

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # train_array_all, test_array_all = get_normalization(data_file, log_file)
    test_feature_map, test_label = generate_feature_maps_wesad(test_array_all, strategy=featureMap_stratgy)
    #
    train_feature_map, train_label = generate_feature_maps_wesad(train_array_all, strategy=featureMap_stratgy)
    if featureMap_stratgy == 'All_concat':
        test_feature_map1, test_label1 = generate_feature_maps_wesad(test_array_all, strategy='AllFromOne')
        test_feature_map2, test_label2 = generate_feature_maps_wesad(test_array_all, strategy='HalfAndHalf')
        test_feature_map3, test_label3 = generate_feature_maps_wesad(test_array_all, strategy='HalfAndRandom')
        #
        train_feature_map1, train_label1 = generate_feature_maps_wesad(train_array_all, strategy='AllFromOne')
        train_feature_map2, train_label2 = generate_feature_maps_wesad(train_array_all, strategy='HalfAndHalf')
        train_feature_map3, train_label3 = generate_feature_maps_wesad(train_array_all, strategy='HalfAndRandom')

        test_feature_map = np.dstack((test_feature_map1, test_feature_map2))
        test_feature_map = np.dstack((test_feature_map, test_feature_map3))
        test_label = np.hstack((test_label1, test_label2))
        test_label = np.hstack((test_label, test_label3))
        train_feature_map = np.dstack((train_feature_map1, train_feature_map2))
        train_feature_map = np.dstack((train_feature_map, train_feature_map3))
        train_label = np.hstack((train_label1, train_label2))
        train_label = np.hstack((train_label, train_label3))

    # Randomly remove one feature for Feature selection

    # transfer feature maps into the correct form (c, d, w, h)
    x_test = feature_map_trans(test_feature_map)
    x_train = feature_map_trans(train_feature_map)
    # Convert the ndarray to a PyTorch tensor
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(np.array(train_label).astype(int))

    # Create a TensorDataset object
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)

    # Create a DataLoader object
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Convert the ndarray to a PyTorch tensor
    x_test_tensor = torch.from_numpy(x_test)
    y_test_tensor = torch.from_numpy(np.array(test_label).astype(int))

    # Create a TensorDataset object
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    # Create a DataLoader object
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    # testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Model
    # print('==> Building model..')

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
    for epoch in range(start_epoch, start_epoch + 20):
        train(epoch)
        now_acc, now_f1 = test(epoch)
        scheduler.step()
        if now_acc > total_best:
            total_best = now_acc
        if now_f1 > total_f1:
            total_f1 = now_f1
    # release memory
    del net, optimizer
    torch.cuda.empty_cache()
    return total_best, total_f1


def get_train_test_rfe_wesad(train_array, test_array):
    Participants_train = set(train_array[-1, :])
    Participants_test = set(test_array[-1, :])
    Participants_list = list(set.union(Participants_train, Participants_test))
    array_all = np.hstack((train_array, test_array))
    # Randomly split participants
    n_total = len(Participants_list)
    offset = round(n_total * 0.8)
    if n_total == 0 or offset < 1:
        return [], Participants_list
    # random split the train and test
    shuffle(Participants_list)
    train_P_list = Participants_list[:offset]
    test_P_list = Participants_list[offset:]
    train_array_resplit = np.zeros((train_array.shape[0], 1))
    test_array_resplit = np.zeros((test_array.shape[0], 1))
    for train_P in train_P_list:
        train_loc = np.where(array_all[-1, :] == train_P)
        train_now_array = array_all[:, train_loc[0]]
        train_array_resplit = np.hstack((train_array_resplit, train_now_array))
    for test_P in test_P_list:
        test_loc = np.where(array_all[-1, :] == test_P)
        test_now_array = array_all[:, test_loc[0]]
        test_array_resplit = np.hstack((test_array_resplit, test_now_array))
    # remove the first 0 line
    train_array_resplit = train_array_resplit[:, 1:]
    test_array_resplit = test_array_resplit[:, 1:]
    return train_array_resplit, test_array_resplit


if __name__ == '__main__':
    classes = ('baseline', 'stress', 'amusement')
    # Decide the feature maps generation strategy:
    # strategy_list = ['AllFromOne', 'HalfAndHalf', 'HalfAndRandom', 'All_concat']
    strategy = 'All_concat'
    # Data
    print('==> Preparing data..')
    log_file = '../log_data_normalization_WESAD_KNN.log'
    data_file = '../json_files'
    train_feature_name_list, test_feature_name_list, train_feature_array, test_feature_array, \
        train_label_array, test_label_array, train_P_label, test_P_label = data_extraction_wesad(data_file)
    train_array_all, test_array_all = get_normalization_wesad(data_file, log_file)
    # Remove features one by one and see the performances
    Over_all_best_acc = 0.0
    Over_all_f1 = 0.0
    Over_all_best_removed_features = []
    min_feature_num = 66
    remained_features = train_feature_name_list.copy()
    removed_features = []
    train_array_rfe = train_array_all.copy()
    test_array_rfe = test_array_all.copy()
    Model_name = ''
    while (train_array_rfe.shape[0] > min_feature_num):
        print('\n\nremained_features num: ')
        print(train_array_rfe.shape[0] - 2)

        # Using Fisher score to decide which part of features should be tested
        # For Fisher score, higher score, stronger ability for improving the performance
        # We just remove one be one of the last 30 features according to Fisher score
        all_array = np.hstack((train_array_rfe, test_array_rfe))
        rank_dic = Feature_selection_fisher_score_wesad(all_array)
        rank_dic_ascend = sorted(rank_dic.items(), key=lambda x: x[1], reverse=False)

        best_acc_rfe = 0
        best_f1_rfe = 0
        removed_feature = ''
        removed_feature_index = 0
        removed_feature_index_list = []
        # for index_feature in range(train_array_rfe.shape[0] - 3):
        # for index_feature in range(3):
        for fisher_index in range(15):
            # We just test the former 30 features with fisher score
            index_feature = rank_dic_ascend[fisher_index][0]
            print('\nNow index feature:')
            print(index_feature)
            print('Removing feature: ' + str(remained_features[index_feature]))

            if index_feature == 0:
                train_array_now = train_array_rfe[index_feature + 1:, :]
                test_array_now = test_array_rfe[index_feature + 1:, :]
            else:
                train_array_now = np.vstack(
                    (train_array_rfe[:index_feature, :], train_array_rfe[index_feature + 1:, :]))
                test_array_now = np.vstack((test_array_rfe[:index_feature, :], test_array_rfe[index_feature + 1:, :]))

            # K-fold cross validation.
            K = 5
            bestAcc_list = []
            bestF1_list = []
            # net = ResNeXt29_2x64d(num_classes=len(classes))
            # net = EfficientNetB0(num_classes=len(classes))
            for cross_index in range(K):
                print('\nStart_Kfold: ' + str(cross_index + 1))
                # net = ResNeXt29_2x64d(num_classes=len(classes))
                net = EfficientNetB0(num_classes=len(classes))
                Model_name = str(net.__class__.__name__)
                # According to cross-validation, every fold should use different train and test
                train_array_K, test_array_K = get_train_test_rfe_wesad(train_array_now, test_array_now)

                best_acc = 0  # best test accuracy
                best_f1 = 0.0  # best test accuracy
                # try:
                #     now_best, now_f1 = training_process_rfe(net, featureMap_stratgy=strategy,
                #                                     train_array_all=train_array_K, test_array_all=test_array_K)
                # except:
                #     net = ResNeXt29_2x64d(num_classes=len(classes), linear_num=4096)
                #     now_best, now_f1 = training_process_rfe(net, featureMap_stratgy=strategy,
                #                                     train_array_all=train_array_K, test_array_all=test_array_K)
                now_best, now_f1 = training_process_rfe(net, featureMap_stratgy=strategy,
                                                        train_array_all=train_array_K, test_array_all=test_array_K)
                bestAcc_list.append(now_best)
                bestF1_list.append(now_f1)
                print('now acc list:' + str(bestAcc_list))
                print('now f1 list:' + str(bestF1_list))
                # Clear the net structure to reduce the influence
                net = None
                del now_best, now_f1, net
                torch.cuda.empty_cache()
            print('\nModel name:' + Model_name)
            # print('Strategy: ' + strategy)
            print('Now removed feature: ' + str(remained_features[index_feature]))
            print('bestAcc_list: ')
            print(bestAcc_list)
            print('Acc std:' + str(np.std(bestAcc_list)))
            print('Mean best acc:' + str(np.mean(bestAcc_list)))
            print('bestF1_list: ')
            print(bestF1_list)
            print('F1 std:' + str(np.std(bestF1_list)))
            print('Mean best F1:' + str(np.mean(bestF1_list)))
            # judge if the acc > the best acc in this rfe loop
            if np.mean(bestAcc_list) > best_acc_rfe:
                best_acc_rfe = np.mean(bestAcc_list)
                best_f1_rfe = np.mean(bestF1_list)
                print('Change to best acc:' + str(best_acc_rfe))
                print('Corresponding f1:' + str(best_f1_rfe))
                removed_feature = remained_features[index_feature]
                removed_feature_index = index_feature
        removed_features.append(removed_feature)
        remained_features.remove(removed_feature)
        # remove that feature from origin
        if removed_feature_index == 0:
            train_array_rfe = train_array_rfe[removed_feature_index + 1:, :]
            test_array_rfe = test_array_rfe[removed_feature_index + 1:, :]
        else:
            train_array_rfe = np.vstack(
                (train_array_rfe[:removed_feature_index, :], train_array_rfe[removed_feature_index + 1:, :]))
            test_array_rfe = np.vstack(
                (test_array_rfe[:removed_feature_index, :], test_array_rfe[removed_feature_index + 1:, :]))

        print('\nRFE best acc:' + str(best_acc_rfe))
        print('\nRFE corresponding f1:' + str(best_f1_rfe))

        if best_acc_rfe > Over_all_best_acc:
            print('Change to over all!')
            Over_all_best_acc = best_acc_rfe
            Over_all_f1 = best_f1_rfe
            Over_all_best_removed_features = removed_features

        print('Over all best acc now:' + str(Over_all_best_acc))
        print('now corresponding f1:' + str(Over_all_f1))

        best_acc_rfe = 0
        print('Removed feature: ' + removed_feature)
        for f_value in removed_features:
            p = train_feature_name_list.index(f_value)
            removed_feature_index_list.append(p)
        print('Removed feature index list: ' + str(removed_feature_index_list))
        print('Removed feature index list amount: ' + str(len(removed_feature_index_list)))
        print('Removed feature list: ' + str(removed_features))
    print('\n\nTraining Process Done!')
    print('Over all best acc final:' + str(Over_all_best_acc))
    print('Corresponding f1 final:' + str(Over_all_f1))
    print('Over all best removed features:' + str(Over_all_best_removed_features))
