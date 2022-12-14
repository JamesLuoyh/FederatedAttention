import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
# print(sys.path)

import glob
import json
import math
import copy
import random
import argparse
from colorama import Fore
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
# import tensorflow as tf

from configs.hyperparameters import *
from trainers.server_fedavg import FedAvg
import datasets.stackoverflow.stackoverflow_nwp_dataloader as stackoverflow_nwp_dataloader
import datasets.emnist.emnist_dataloader as emnist_dataloader
import datasets.cifar10.cifar10_dataloader as cifar10_dataloader
from models.emnist_cnn import EMNISTNet
from models.cifar100_resnet import resnet18 as CifarResNet
from models.stackoverflow_net import StackoverflowNet

AVAILABLE_GPUS = 1 #torch.cuda.device_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(56)
torch.manual_seed(56)
np.random.seed(56)
def stackoverflow_nwp_train(net, train_data, test_data):
    original_net = copy.deepcopy(net)

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
    optimizer = optim.SGD(original_net.parameters(), lr=lr, momentum=0.9)
    
    max_accuracy = 0.0
    for epoch in range(epochs):
        original_net.train()
        accumulated_loss = 0
        sample_count = 0
        
        for tokens, next_tokens in train_data:
            tokens_, next_tokens_ = torch.from_numpy(tokens.numpy()).to(DEVICE), torch.from_numpy(next_tokens.numpy()).to(DEVICE)
            optimizer.zero_grad()
            logits = original_net(tokens_)
            loss = criterion(logits, next_tokens_)
            accumulated_loss += loss.item()
            sample_count += tokens.shape[0]
            loss.backward()
            optimizer.step()
    
        _, accuracy, _, _ = stackoverflow_nwp_test(original_net, test_data)
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            for n, o in zip(net.parameters(), original_net.parameters()):
                n.data = o.data

    with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'a+') as f:
        f.write(str(max_accuracy)+"\n")

    return accumulated_loss / sample_count, sample_count

def stackoverflow_nwp_test(net, test_data):
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for tokens, next_tokens in test_data:
            tokens_, next_tokens_ = torch.from_numpy(tokens.numpy()).to(DEVICE), torch.from_numpy(next_tokens.numpy()).to(DEVICE)
            logits = net(tokens_)
            loss += criterion(logits, next_tokens_).item()
            _, predicted = torch.max(logits, 1)
            paddings = ~(next_tokens_ == 0)
            total += torch.count_nonzero(paddings).item()
            correct += ((predicted == next_tokens_) * paddings).sum().item()

    accuracy = correct / total
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return loss, accuracy, correct, total

def stackoverflow_nwp_generalized_test(net, train_data, test_data):
    original_net = copy.deepcopy(net)

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
    optimizer = optim.SGD(original_net.parameters(), lr=lr, momentum=0.9)

    before_loss, before_accuracy, before_correct, before_total = stackoverflow_nwp_test(original_net, test_data)
    after_loss, after_accuracy, after_correct, after_total = 0.0, 0.0, 0, 0

    original_net.train()
    for epoch in range(epochs):
        
        for tokens, next_tokens in train_data:
            tokens_, next_tokens_ = torch.from_numpy(tokens.numpy()).to(DEVICE), torch.from_numpy(next_tokens.numpy()).to(DEVICE)
            optimizer.zero_grad()
            logits = net(tokens_)
            loss = criterion(logits, next_tokens_)
            loss.backward()
            optimizer.step()

        loss, accuracy, correct, total = stackoverflow_nwp_test(original_net, test_data)
        if after_accuracy < accuracy:
            after_loss, after_accuracy, after_correct, after_total = loss, accuracy, correct, total
    
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return before_loss, before_accuracy, after_loss, after_accuracy, after_correct, after_total

def emnist_train(net, train_data, test_data, model_dim):
    original_net = copy.deepcopy(net)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(original_net.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # optimizer = optim.Adam(original_net.parameters(), lr=lr)
    max_epoch = 0
    max_accuracy = 0.0
    
    client_embeddings = torch.zeros(128).to(DEVICE)
    data_size = 0
    final_grads = np.zeros(model_dim)
    for epoch in range(epochs):
        original_net.train()
        
        accumulated_loss = 0
        sample_count = 0
        accumulated_embeddings = torch.zeros(128).to(DEVICE)
        aggregated_grads = np.zeros(model_dim)
        for inputs, labels in train_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).type(torch.LongTensor).to(DEVICE)
            # if inputs.shape[0] == 1:
            #     break
            data_size += inputs.shape[0]
            optimizer.zero_grad()
            logits, embeddings = original_net(inputs)
            accumulated_embeddings += torch.sum(embeddings, dim=0)
            
            loss = criterion(logits, labels)
            # print(loss)
            accumulated_loss += loss.item()
            sample_count += inputs.shape[0]
            loss.backward()
            grads = []
            bias = False
            for param in original_net.parameters():
                # Weights and bias alternate in the params
                if bias:
                    grads.append(param.grad.view(-1))
                bias = not bias
            grads = torch.cat(grads)
            aggregated_grads += grads.detach().cpu().numpy()
            optimizer.step()
    
        _, accuracy, _, _ = emnist_test(original_net, test_data)
        if max_accuracy <= accuracy:
            # print(1)
            max_accuracy = accuracy
            max_epoch = epoch
            for n, o in zip(net.parameters(), original_net.parameters()):
                n.data = o.data
                
            client_embeddings = accumulated_embeddings
            final_grads = aggregated_grads

    # print('9')
    with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'a+') as f:
        f.write(str(max_accuracy)+"\n")
    # print('10')
    # print("data_size", data_size)
    return accumulated_loss / sample_count, sample_count, max_epoch, client_embeddings.detach().cpu().numpy() / data_size, final_grads / data_size

def emnist_test(net, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for inputs, labels in test_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).type(torch.LongTensor).to(DEVICE)
            # if inputs.shape[0] == 1:
            #     break
            logits, embedding = net(inputs)
            loss += criterion(logits, labels).item()
            _, predicted = torch.max(logits, 1)
            total += labels.shape[0]
            correct += ((predicted == labels)).sum().item()

    accuracy = correct / total
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return loss, accuracy, correct, total

def emnist_generalized_test(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    before_loss, before_accuracy, before_correct, before_total = emnist_test(net, test_data)
    after_loss, after_accuracy, after_correct, after_total = 0.0, 0.0, 0, 0

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0) #
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        net.train()
        
        for inputs, labels in train_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).type(torch.LongTensor).to(DEVICE)
            # if inputs.shape[0] == 1:
            #     break
            optimizer.zero_grad()
            try:
                logits, embedding = net(inputs)
            except:
                print(inputs.shape)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        loss, accuracy, correct, total = emnist_test(net, test_data)
        if after_accuracy <= accuracy:
            after_loss, after_accuracy, after_correct, after_total = loss, accuracy, correct, total
    # print("Accuracy: ", correct, "/", total, " = ", accuracy)
    return before_loss, before_accuracy, after_loss, after_accuracy, after_correct, after_total

def cifar10_train(net, train_data, test_data):
    original_net = copy.deepcopy(net)
    
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(original_net.parameters(), lr=lr, momentum=0.9)
    
    max_epoch = 0
    max_accuracy = 0.0
    for epoch in range(epochs):
        original_net.train()
        
        accumulated_loss = 0
        sample_count = 0
        for inputs, labels in train_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).to(DEVICE)
            inputs = inputs.swapaxes(1,3)
            inputs = inputs.swapaxes(2,3)

            optimizer.zero_grad()
            logits = original_net(inputs)
            loss = criterion(logits, labels)
            accumulated_loss += loss.item()
            sample_count += inputs.shape[0]
            loss.backward()
            optimizer.step()
    
        _, accuracy, _, _ = cifar10_test(original_net, test_data)
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            max_epoch = epoch
            for n, o in zip(net.parameters(), original_net.parameters()):
                n.data = o.data

    with open(checkpoint_dir + 'intermediate_'+METHOD+ '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(total_rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_test_accuracies.txt', 'a+') as f:
        f.write(str(max_accuracy)+"\n")

    return accumulated_loss / sample_count, sample_count, max_epoch

def cifar10_test(net, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for inputs, labels in test_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).to(DEVICE)
            inputs = inputs.swapaxes(1,3)
            inputs = inputs.swapaxes(2,3)

            logits = net(inputs)
            loss += criterion(logits, labels).item()
            _, predicted = torch.max(logits, 1)
            total += labels.shape[0]
            correct += ((predicted == labels)).sum().item()

    accuracy = correct / total
    return loss, accuracy, correct, total

def cifar10_generalized_test(net, train_data, test_data):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    before_loss, before_accuracy, before_correct, before_total = cifar10_test(net, test_data)
    after_loss, after_accuracy, after_correct, after_total = 0.0, 0.0, 0, 0

    for epoch in range(epochs):
        net.train()
        
        for inputs, labels in train_data:
            labels = labels.numpy()
            inputs, labels = torch.from_numpy(inputs.numpy()).to(DEVICE), torch.from_numpy(labels).to(DEVICE)
            inputs = inputs.swapaxes(1,3)
            inputs = inputs.swapaxes(2,3)
            
            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
        loss, accuracy, correct, total = cifar10_test(net, test_data)
        if after_accuracy < accuracy:
            after_loss, after_accuracy, after_correct, after_total = loss, accuracy, correct, total
    
    return before_loss, before_accuracy, after_loss, after_accuracy, after_correct, after_total


class StackoverflowNWPClient(fl.client.NumPyClient):

    def __init__(self, cid, net):# -> None:
        self.net = net
        self.cid = cid.strip()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        train_data_ = train_data.create_tf_dataset_for_client(self.cid)
        test_data_ = test_data.create_tf_dataset_for_client(self.cid)
        self.set_parameters(parameters)
        loss, count = stackoverflow_nwp_train(self.net, train_data_, test_data_)
        return self.get_parameters(config), count, {'loss': float(loss), 'gradient_alpha': float(0.0), 'gradient_personalized_objective': float(0.0)}

    def evaluate(self, parameters, config):
        train_data_ = train_data.create_tf_dataset_for_client(self.cid)
        test_data_ = test_data.create_tf_dataset_for_client(self.cid)
        self.set_parameters(parameters)
        # loss, accuracy, correct, count = stackoverflow_test(self.net, test_data_)
        # return float(loss), count, {'accuracy': float(accuracy), 'correct': int(correct)}
        before_loss, before_accuracy, after_loss, after_accuracy, correct, count = stackoverflow_nwp_generalized_test(self.net, train_data_, test_data_)
        return float(after_loss), count, {'before_accuracy': float(before_accuracy), 'after_accuracy': float(after_accuracy), 'correct': int(correct)}

class EMNISTClient(fl.client.NumPyClient):

    def __init__(self, cid, net):# -> None:
        self.net = net
        self.cid = cid.strip()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        train_data_ = train_data.create_tf_dataset_for_client(self.cid)
        test_data_ = test_data.create_tf_dataset_for_client(self.cid)
        # parameters = config['params']
        # print(parameters)
        model_dim = config['model_dim']
        # print(model_dim)
        self.set_parameters(parameters)
        loss, count, max_epoch, client_emb, grads = emnist_train(self.net, train_data_, test_data_, model_dim)
        print('b')
        return self.get_parameters(config), count, {'loss': float(loss), 'max_epoch': max_epoch, 'gradient_alpha': float(0.0), 'gradient_personalized_objective': float(0.0), 'client_embedding': client_emb, 'grads': grads}

    def evaluate(self, parameters, config):
        train_data_ = train_data.create_tf_dataset_for_client(self.cid)
        test_data_ = test_data.create_tf_dataset_for_client(self.cid)
        # parameters = config['params']
        self.set_parameters(parameters)
        # loss, accuracy, correct, count = stackoverflow_test(self.net, test_data_)
        # return float(loss), count, {'accuracy': float(accuracy), 'correct': int(correct)}
        before_loss, before_accuracy, after_loss, after_accuracy, correct, count = emnist_generalized_test(self.net, train_data_, test_data_)
        return float(after_loss), count, {'before_accuracy': float(before_accuracy), 'after_accuracy': float(after_accuracy), 'correct': int(correct)}

class Cifar10Client(fl.client.NumPyClient):

    def __init__(self, cid, net):# -> None:
        self.net = net
        self.cid = str(cid)#.strip()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        named_params = [n for n, p in self.net.named_parameters()]
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict if k in named_params})
        self.net.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        train_data_ = train_data.create_tf_dataset_for_client(self.cid)
        test_data_ = test_data.create_tf_dataset_for_client(self.cid)
        self.set_parameters(parameters)
        loss, count, max_epoch = cifar10_train(self.net, train_data_, test_data_)
        return self.get_parameters(), count, {'loss': float(loss), 'max_epoch': max_epoch, 'gradient_alpha': float(0.0), 'gradient_personalized_objective': float(0.0)}

    def evaluate(self, parameters, config):
        train_data_ = train_data.create_tf_dataset_for_client(self.cid)
        test_data_ = test_data.create_tf_dataset_for_client(self.cid)
        self.set_parameters(parameters)
        before_loss, before_accuracy, after_loss, after_accuracy, correct, count = cifar10_generalized_test(self.net, train_data_, test_data_)
        return float(after_loss), count, {'before_accuracy': float(before_accuracy), 'after_accuracy': float(after_accuracy), 'correct': int(correct)}


def client_fn(cid: str):# -> fl.client.Client:
    # print('Picked client #', cid)
    if DATASET == 'stackoverflow_nwp':
        net = StackoverflowNet(vocab_size + 4, embedding_size, hidden_size, num_layers).to(DEVICE)
        return StackoverflowNWPClient(cid, net)
    elif DATASET == 'emnist':
        net = EMNISTNet().to(DEVICE)
        return EMNISTClient(cid, net)
    elif DATASET == 'cifar10':
        net = CifarResNet().to(DEVICE)
        return Cifar10Client(cid, net)

def load_parameters_from_disk():# -> fl.common.Parameters:
    model_file_name = checkpoint_dir + 'model_' + METHOD + '_' + str(num_clients) + '_' + str(total_clients) + '_' + str(rounds) + '_' + str(epochs) + '_' + str(lr).replace('.', '_') + '_' + '.pth'
    if not os.path.exists(model_file_name):
        return None, rounds
    print("Loading: ", model_file_name)

    checkpoint = torch.load(model_file_name)

    if DATASET == 'stackoverflow_nwp':
        net = StackoverflowNet(vocab_size + 4, embedding_size, hidden_size, num_layers).to(DEVICE)
    elif DATASET == 'emnist':
        net = EMNISTNet().to(DEVICE)
    elif DATASET == 'cifar10':
        net = CifarResNet().to(DEVICE)
    
    net.load_state_dict(checkpoint['net_state_dict'])
    print(Fore.YELLOW + f"Loading model weights from round #{checkpoint['round']}" + Fore.WHITE)
    
    return fl.common.weights_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()]), rounds - checkpoint['round']

if __name__  == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help = 'Enter the dataset you want to train your algorithm on.')
    args = parser.parse_args()

    DATASET = args.dataset
    METHOD = 'FedAvg'

    if not (DATASET in ['stackoverflow_nwp', 'emnist', 'cifar10']):
        print('Dataset not recognized, try again!')
        sys.exit()    
    hyperparameters = eval(DATASET + '_' + METHOD.lower()) 
    
    num_clients = hyperparameters['num_clients']
    total_clients = hyperparameters['total_clients']
    rounds = hyperparameters['rounds']
    total_rounds = hyperparameters['rounds']
    epochs = hyperparameters['epochs']
    checkpoint_interval = hyperparameters['checkpoint_interval']
    checkpoint_dir = hyperparameters['checkpoint_dir']
    dataset_dir = hyperparameters['dataset_dir']
    batch_size = hyperparameters['batch_size']
    lr = hyperparameters['lr']

    if DATASET == 'stackoverflow_nwp':
        vocab_size = hyperparameters['vocab_size']
        embedding_size = hyperparameters['embedding_size']
        hidden_size = hyperparameters['hidden_size']
        num_layers = hyperparameters['num_layers']
        sequence_length = hyperparameters['sequence_length']
        train_data, test_data = stackoverflow_nwp_dataloader.get_federated_datasets(vocab_size, sequence_length, train_client_batch_size=batch_size)
        with open(dataset_dir + 'available_clients.txt') as f:
            clients = f.readlines()[:total_clients]

    elif DATASET == 'emnist':
        train_data, test_data = emnist_dataloader.get_federated_datasets(train_client_batch_size=batch_size)
        with open(dataset_dir + 'available_train_clients.txt', 'r') as f:
            clients = f.readlines()[:50]

    elif DATASET == 'cifar10':
        train_data, test_data = cifar10_dataloader.get_federated_datasets()
        clients = range(total_clients)

    initial_parameters, rounds = load_parameters_from_disk()

    strategy = FedAvg(
        clients,
        fraction_fit=num_clients / len(clients), 
        fraction_eval=num_clients / len(clients), 
        min_fit_clients=num_clients,
        min_eval_clients=num_clients,
        min_available_clients=num_clients,
        dataset=DATASET,
        client_algorithm=METHOD,
        initial_parameters=initial_parameters
    )

    print(Fore.RED + "Availble Device: " + str(DEVICE) + ", Count: " + str(AVAILABLE_GPUS) + Fore.WHITE)

    fl.simulation.start_simulation(
        client_fn = client_fn,
        client_resources = {'num_gpus': AVAILABLE_GPUS},
        clients_ids = clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy = strategy,
        ray_init_args = {'num_gpus': AVAILABLE_GPUS}
    )

    ## Stackoverflow Test - Single Client
    # train_data_ = train_data.create_tf_dataset_for_client('06580021')
    # test_data_ = test_data.create_tf_dataset_for_client('06580021')
    # net = StackoverflowNet(vocab_size + 4, embedding_size, hidden_size, num_layers).to(DEVICE)
    # # client = StackoverflowClient('06580021', net)
    # # accuracy, count = client.fit()
    # accuracy, count = stackoverflow_train(net, train_data_)
    # print("Accuracy: ", accuracy, ", Count: ", count)
    # # loss, accuracy, correct, count = client.evaluate()
    # loss, accuracy, correct, count = stackoverflow_test(net, test_data_)
    # print("Loss: ", loss, ", Accuracy: ", accuracy, ", Count: ", count)
   
