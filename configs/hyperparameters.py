stackoverflow_nwp_fedavg = {
    'num_clients': 10,
    'total_clients': 195303,
    'rounds': 2000,
    'epochs': 10,
    'num_layers': 1,
    'batch_size': 16,
    'vocab_size': 10000,
    'hidden_size': 670,
    'sequence_length': 20,
    'embedding_size': 96,
    'lr': 0.3,
    'checkpoint_dir': './results/stackoverflow_nwp/FedAvg/',
    'dataset_dir': './datasets/stackoverflow/',
    'checkpoint_interval': 10,
}

emnist_fedavg = {
    'num_clients': 10,
    'total_clients': 3400,
    'rounds': 1500,
    'epochs': 1, #1,
    'num_layers': 1,
    'batch_size': 20,
    'lr': 0.0001,
    'checkpoint_dir': './results/emnist/FedAttn/',
    'dataset_dir': './datasets/emnist/',
    'checkpoint_interval': 10,
}

cifar10_fedavg = {
    'num_clients': 10,
    'total_clients': 500,
    'rounds': 4000,
    'epochs': 10, #1,
    'batch_size': 20,
    'lr': 0.005,
    'checkpoint_dir': './results/cifar10/FedAvg/',
    'dataset_dir': './datasets/cifar10/',
    'checkpoint_interval': 10,
}