import glob
import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import json
import math
import torch
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from colorama import Fore
from typing import Optional, List, OrderedDict, Tuple, Dict
from flwr.common import Parameters, Scalar, MetricsAggregationFn, FitIns, EvaluateIns, FitRes, EvaluateRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from models.stackoverflow_net import StackoverflowNet
from models.emnist_cnn import EMNISTNet

from configs.hyperparameters import *

import faiss 

class FedAvg(fl.server.strategy.FedAvg):

    def __init__(
        self,
        clients,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        dataset: str = 'stackoverflow',
        client_algorithm: str = 'FedAvg',
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.dataset_name = dataset
        self.dataset_hyperparameters = eval(str(dataset)+'_'+str(client_algorithm).lower())
        self.client_algorithm = client_algorithm

        self.lr = self.dataset_hyperparameters['lr']
        self.epochs = self.dataset_hyperparameters['epochs']
        if client_algorithm == 'PerFedAvg':
            self.lr = self.dataset_hyperparameters['local_lr']
            self.epochs = self.dataset_hyperparameters['local_epochs']
        
        self.total_rounds = self.dataset_hyperparameters['rounds']
        self.num_clients = self.dataset_hyperparameters['num_clients']
        self.total_clients = self.dataset_hyperparameters['total_clients']
        self.checkpoint_dir = self.dataset_hyperparameters['checkpoint_dir']
        self.checkpoint_interval = self.dataset_hyperparameters['checkpoint_interval']

        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        self.model_file_name = self.checkpoint_dir + 'model_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.pth'
        self.loss_file_name = self.checkpoint_dir + 'losses_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.before_accu_file_name = self.checkpoint_dir + 'before_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.after_accu_file_name = self.checkpoint_dir + 'after_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        
        self.all_loss_file_name = self.checkpoint_dir + 'all_losses_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.all_before_accu_file_name = self.checkpoint_dir + 'all_before_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.all_after_accu_file_name = self.checkpoint_dir + 'all_after_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        
        self.plot_loss_file_name = self.checkpoint_dir + 'plot_losses_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'
        self.plot_before_accu_file_name = self.checkpoint_dir + 'plot_before_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'
        self.plot_after_accu_file_name = self.checkpoint_dir + 'plot_after_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'
        
        self.personalized_accu_file_name = self.checkpoint_dir + 'personalized_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.personalized_all_accu_file_name = self.checkpoint_dir + 'all_personalized_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.personalized_plot_accu_file_name = self.checkpoint_dir + 'plot_personalized_accuracies_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'
        
        self.gradient_beta_norm_file_name = self.checkpoint_dir + 'gradient_beta_norm_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.gradient_alpha_clients_norm_file_name = self.checkpoint_dir + 'gradient_alpha_clients_norm_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.gradient_personalized_objective_clients_norm_file_name = self.checkpoint_dir + 'gradient_personalized_objective_clients_norm_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        self.gradient_delta_clients_norm_file_name = self.checkpoint_dir + 'gradient_delta_clients_norm_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
        
        # self.plot_gradient_norm_file_name = self.checkpoint_dir + 'gradient_norm_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'

        if self.dataset_name == 'movielens':
            self.test_loss_file_name = self.checkpoint_dir + 'test_losses_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
            self.test_error_loss_file_name = self.checkpoint_dir + 'error_test_losses_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.json'
            self.plot_test_loss_file_name = self.checkpoint_dir + 'plot_test_losses_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'
            self.plot_error_test_loss_file_name = self.checkpoint_dir + 'plot_error_test_losses_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '.png'
            self.test_losses_dict = {}
        
        self.losses_dict = {}
        self.before_accuracies_dict = {}
        self.after_accuracies_dict = {}
        self.all_losses_dict = {}
        self.all_before_accuracies_dict = {}
        self.all_after_accuracies_dict = {}
        self.personalized_accuracies_dict = {}
        self.all_personalized_accuracies_dict = {}
        self.gradient_beta_norms_dict = {}
        self.gradient_alpha_clients_norms_dict = {}
        self.gradient_personalized_objective_clients_norms_dict = {}
        self.gradient_delta_clients_norms_dict = {}
        self.round_offset = 0

        self.net = EMNISTNet()
        self.ids = {}
        self.counter = 0
        self.embed_dim = 128

        try:
            os.remove(self.checkpoint_dir + 'intermediate_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '_test_accuracies.txt')
        except:
            pass
        
        if os.path.exists(self.loss_file_name):
            with open(self.loss_file_name, 'r') as f:
                losses_raw = f.read()
            self.losses_dict = json.loads(losses_raw)

            with open(self.all_loss_file_name, 'r') as f:
                all_losses_raw = f.read()
            self.all_losses_dict = json.loads(all_losses_raw)

            with open(self.before_accu_file_name, 'r') as f:
                accuracies_raw = f.read()
            self.before_accuracies_dict = json.loads(accuracies_raw)
            
            with open(self.after_accu_file_name, 'r') as f:
                accuracies_raw = f.read()
            self.after_accuracies_dict = json.loads(accuracies_raw)

            with open(self.all_before_accu_file_name, 'r') as f:
                all_accuracies_raw = f.read()
            self.all_before_accuracies_dict = json.loads(all_accuracies_raw)

            with open(self.all_after_accu_file_name, 'r') as f:
                all_accuracies_raw = f.read()
            self.all_after_accuracies_dict = json.loads(all_accuracies_raw)

            with open(self.personalized_accu_file_name, 'r') as f:
                personalized_accuracies_raw = f.read()
            self.personalized_accuracies_dict = json.loads(personalized_accuracies_raw)

            with open(self.personalized_all_accu_file_name, 'r') as f:
                all_personalized_accuracies_raw = f.read()
            self.all_personalized_accuracies_dict = json.loads(all_personalized_accuracies_raw)

            with open(self.gradient_beta_norm_file_name, 'r') as f:
                gradient_beta_norm_raw = f.read()
            self.gradient_beta_norms_dict = json.loads(gradient_beta_norm_raw)
            
            with open(self.gradient_alpha_clients_norm_file_name, 'r') as f:
                gradient_alpha_clients_norm_raw = f.read()
            self.gradient_alpha_clients_norms_dict = json.loads(gradient_alpha_clients_norm_raw)
            
            with open(self.gradient_personalized_objective_clients_norm_file_name, 'r') as f:
                gradient_personalized_objective_clients_norm_raw = f.read()
            self.gradient_personalized_objective_clients_norms_dict = json.loads(gradient_personalized_objective_clients_norm_raw)
            
            with open(self.gradient_delta_clients_norm_file_name, 'r') as f:
                gradient_delta_clients_norm_raw = f.read()
            self.gradient_delta_clients_norms_dict = json.loads(gradient_delta_clients_norm_raw)
            
            if self.dataset_name == 'movielens':
                with open(self.test_loss_file_name, 'r') as f:
                    test_losses_raw = f.read()
                self.test_losses_dict = json.loads(test_losses_raw)

            self.round_offset = max([int(i) for i in self.losses_dict.keys()])

        self.build_ann_index()

        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_eval, 
            min_fit_clients=min_fit_clients, min_evaluate_clients=min_eval_clients, 
            min_available_clients=min_available_clients, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)

    def build_ann_index(self):
        res = faiss.StandardGpuResources()
        values = self.net.state_dict().values()
        values = [value.numpy().flatten() for value in values]
        self.d = len(np.concatenate(values))                       # dimension
        index = faiss.IndexFlatL2(self.embed_dim)   # build the index
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        self.ann_index = faiss.IndexIDMap2(index)
        self.attention = AttnModel(self.embed_dim, self.d, n_head=1, drop_out=0.1)#.to(DEVICE)
        # self.attention = ScaledAttention(np.power(1000, 0.5))
        self.params_dict = np.zeros((self.total_clients, self.d))
        # self.optimizer = optim.SGD(self.attention.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.attention.parameters(), lr=0.0001, weight_decay=1e-1)
        self.n_neighbors = 5
        
    def build_params_from_neighbors(self, client_proxy_and_fitins, server_round, parameters):
        cids = []
        for i in client_proxy_and_fitins:
            i[1].config['rnd'] = server_round
            i[1].config['model_dim'] = self.d
            cids.append(i[0].cid)
        self.current_cids = cids
        queries = [self.ann_index.reconstruct(self.ids[cid]) if cid in self.ids else np.zeros(self.embed_dim) for cid in cids]
        queries = np.stack(queries, axis=0).astype("float32")
        D, N = self.ann_index.search(queries, self.n_neighbors)
        # print(D)
        # print(queries, D, N)
        queries_t = torch.from_numpy(queries) # [N_clients, embedding_size]
        queries_t = queries_t.unsqueeze(1) # [N_clients, 1, embedding_size]
        
        neighbor_params = []
        neighbor_embeds = []
        params_ndarrays = fl.common.parameters_to_ndarrays(parameters)
        global_params = [i.flatten() for i in params_ndarrays]
        global_params = np.concatenate(global_params)
        
        for i in range(len(N)): # index >= 0 and distance < 10
            neighbor_embeds.append([self.ann_index.reconstruct(int(index)) if index >= 0 and distance < 10 else np.zeros(self.embed_dim) for index, distance in zip(N[i], D[i])])
            neighbor_params.append([self.params_dict[index] if index >= 0 and distance < 10 else global_params for index, distance in zip(N[i], D[i])])
        neighbor_t = torch.FloatTensor(np.stack(neighbor_embeds, axis=0)) # [N_clients, N_neighbors, embed_size]
        params_t = torch.FloatTensor(np.stack(neighbor_params, axis=0)) # [N_clients, N_neighbors, model_size]
        # print(neighbor_t)
        reconstructed_param, _ = self.attention(queries_t, neighbor_t, params_t, torch.FloatTensor(global_params)) # [N_client, 1, model_size]
        self.reconstructed_param = reconstructed_param
        detached_param = reconstructed_param.detach().numpy()
        for i in range(len(client_proxy_and_fitins)):
            client_params = detached_param[i,0]
            idx = 0
            layers = []
            for j in params_ndarrays:
                layer = client_params[idx: idx + len(j.flatten())]
                layers.append(layer.reshape(j.shape))
                idx += len(j.flatten())
            client_proxy_and_fitins[i][1].config['params'] = layers
        return client_proxy_and_fitins
        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):# -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        print("server_round", server_round)
        client_proxy_and_fitins = super().configure_fit(server_round, parameters, client_manager)
        
        
        return self.build_params_from_neighbors(client_proxy_and_fitins, server_round, parameters)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):# -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        client_proxy_and_fitins = super().configure_evaluate(server_round, parameters, client_manager)
  
        client_proxy_and_fitins = self.build_params_from_neighbors(client_proxy_and_fitins, server_round, parameters)

    # if self.client_algorithm == 'APFL' or self.client_algorithm == 'PerFedAvg':
        with open(self.checkpoint_dir + 'intermediate_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '_test_accuracies.txt') as f:
            accuracies = f.readlines()
            accuracies = [float(acc.strip()) for acc in accuracies]
            
            accuracy_aggregated = sum(accuracies) / len(accuracies)
            print(Fore.RED + f"{self.client_algorithm}: Round {server_round + self.round_offset} personalized accuracy aggregated from client results: {accuracy_aggregated}" + Fore.WHITE)
            os.remove(self.checkpoint_dir + 'intermediate_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '_test_accuracies.txt')
            
            self.personalized_accuracies_dict[int(server_round + self.round_offset)] = accuracy_aggregated
            self.all_personalized_accuracies_dict[int(server_round + self.round_offset)] = accuracies

        if (server_round + self.round_offset) % self.checkpoint_interval == 0:
            with open(self.personalized_accu_file_name, 'w') as f:
                json.dump(self.personalized_accuracies_dict, f)

            with open(self.personalized_all_accu_file_name, 'w') as f:
                json.dump(self.all_personalized_accuracies_dict, f)

            plt.figure(0)
            plt.plot(self.personalized_accuracies_dict.keys(), self.personalized_accuracies_dict.values(), c='#2978A0', label='Average')
            # plt.plot(*zip(*sorted(self.losses_dict.items())), c='#007ba7')
            # mins = [min(vals) for vals in self.all_personalized_accuracies_dict.values()]
            # maxs = [max(vals) for vals in self.all_personalized_accuracies_dict.values()]
            # plt.fill_between(self.all_personalized_accuracies_dict.keys(), mins, maxs, color= '#ccf1ff')
            # percentile25 = [sorted(vals)[2] for vals in self.all_personalized_accuracies_dict.values()]
            # percentile75 = [sorted(vals)[-3] for vals in self.all_personalized_accuracies_dict.values()]
            # plt.plot(self.all_personalized_accuracies_dict.keys(), percentile25, c='#694873', label='25th %ile')
            # plt.plot(self.all_personalized_accuracies_dict.keys(), percentile75, c='#F5F749', label='75th %ile')
            # plt.legend()
            plt.xlabel('Number of Rounds')
            plt.ylabel('Personalized Accuracy')
            plt.title('Validation Accuracy for ' + str(self.client_algorithm) + ' Baseline')
            plt.savefig(self.personalized_plot_accu_file_name)

        return client_proxy_and_fitins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):# -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        print(losses)
        examples = [r.num_examples for _, r in results]

        ##########
        
        #### Use output of the first linear layer as embedding
           
        client_embeddings = []
        cids_for_embed = []
        params_flatten = []
        for c, r in results:
            embedding = r.metrics["client_embedding"]
            if np.sum(embedding) == 0:
                continue
            client_embeddings.append(embedding)
            cids_for_embed.append(c.cid)
            p = fl.common.parameters_to_ndarrays(r.parameters)
            p = [i.flatten() for i in p]
            params_flatten.append(np.concatenate(p))
        if len(client_embeddings) > 0:
            for c in cids_for_embed:
                if c not in self.ids:
                    self.ids[c] = self.counter
                    self.counter += 1

            cids = [self.ids[c] for c in cids_for_embed]
            cids = np.array(cids, dtype='int64')
            client_embeddings = np.stack(client_embeddings, axis=0)
            params_flatten = np.stack(params_flatten, axis=0)
            self.ann_index.remove_ids(cids)
            # print(client_embeddings)
            
            self.ann_index.add_with_ids(client_embeddings, cids)
            self.params_dict[cids] = params_flatten

        #### Do backprop with new gradients
        # cids self.current_cids = 
        
        
        grads_unordered = [r.metrics["grads"] for _, r in results]
        complete_cid = [c.cid for c, _ in results]
        grads = []
        for i in self.current_cids:
            index = complete_cid.index(i)
            grads.append(grads_unordered[index])
        
        grads = torch.from_numpy(np.stack(grads))#.to(DEVICE)
        grads = grads.unsqueeze(1)
        self.optimizer.zero_grad()
        self.reconstructed_param.backward(gradient=grads)
        self.optimizer.step()
        ###########

        # Aggregate and print custom metric
        loss_aggregated = sum(losses) / sum(examples)
        print("")
        print(Fore.BLUE + f"{self.client_algorithm}: Round {server_round + self.round_offset} loss aggregated from client results: {loss_aggregated}" + Fore.WHITE)

        self.losses_dict[int(server_round + self.round_offset)] = loss_aggregated

        all_losses = [r.metrics["loss"] for _, r in results]
        self.all_losses_dict[int(server_round + self.round_offset)] = all_losses

        all_gradient_alpha_clients = [r.metrics['gradient_alpha'] for _, r in results]
        self.gradient_alpha_clients_norms_dict[int(server_round + self.round_offset)] = all_gradient_alpha_clients

        all_gradient_personalized_objective_clients = [r.metrics['gradient_personalized_objective'] for _, r in results]
        self.gradient_personalized_objective_clients_norms_dict[int(server_round + self.round_offset)] = all_gradient_personalized_objective_clients
        
        aggregated_parameters_tuple = super().aggregate_fit(server_round, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple
        aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

        all_delta = []
        for _, r in results:
            weights = fl.common.parameters_to_ndarrays(r.parameters)
            grad_norm = 0.0
            for client_weight, aggregated_weight in zip(weights, aggregated_weights):
                grad_norm += np.linalg.norm(client_weight - aggregated_weight) ** 2
            grad_norm = grad_norm ** (1. / 2)
            all_delta.append(grad_norm)
        self.gradient_delta_clients_norms_dict[int(server_round + self.round_offset)] = all_delta

        if (server_round + self.round_offset) % self.checkpoint_interval == 0:
            with open(self.loss_file_name, 'w') as f:
                json.dump(self.losses_dict, f)

            with open(self.all_loss_file_name, 'w') as f:
                json.dump(self.all_losses_dict, f)

            with open(self.gradient_beta_norm_file_name, 'w') as f:
                json.dump(self.gradient_beta_norms_dict, f)

            with open(self.gradient_alpha_clients_norm_file_name, 'w') as f:
                json.dump(self.gradient_alpha_clients_norms_dict, f)

            with open(self.gradient_personalized_objective_clients_norm_file_name, 'w') as f:
                json.dump(self.gradient_personalized_objective_clients_norms_dict, f)

            with open(self.gradient_delta_clients_norm_file_name, 'w') as f:
                json.dump(self.gradient_delta_clients_norms_dict, f)

            plt.figure(1)
            # plt.plot(*zip(*sorted(self.losses_dict.items())), c='#007ba7')
            plt.plot(self.losses_dict.keys(), self.losses_dict.values(), c='#2978A0', label='Average')
            # mins = [min(vals) for vals in self.all_losses_dict.values()]
            # maxs = [max(vals) for vals in self.all_losses_dict.values()]
            # plt.fill_between(self.all_losses_dict.keys(), mins, maxs, color= '#ccf1ff')
            # percentile25 = [sorted(vals)[2] for vals in self.all_losses_dict.values()]
            # percentile75 = [sorted(vals)[-3] for vals in self.all_losses_dict.values()]
            # plt.plot(self.all_losses_dict.keys(), percentile25, c='#694873', label='25th %ile')
            # plt.plot(self.all_losses_dict.keys(), percentile75, c='#F5F749', label='75th %ile')
            # plt.legend()
            plt.xlabel('Number of Rounds')
            plt.ylabel('Training Loss')
            plt.title('Training Loss for ' + str(self.client_algorithm) + ' Baseline')
            plt.savefig(self.plot_loss_file_name)

            # plt.figure(5)
            # plt.plot(self.gradient_norms_dict.keys(), self.gradient_norms_dict.values(), c='#2978A0', label='Average')
            # plt.xlabel('Number of Rounds')
            # plt.ylabel('Gradient Norm')
            # plt.title('Average gradient norm for ' + str(self.client_algorithm) + ' Baseline')
            # plt.savefig(self.plot_gradient_norm_file_name)

        if (server_round + self.round_offset) == self.total_rounds:
            i = 0
            while os.path.exists(self.loss_file_name[:-5]+"_exp_%s.json" %i):
                i += 1

            os.rename(self.loss_file_name, str(self.loss_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.all_loss_file_name, str(self.all_loss_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.plot_loss_file_name, str(self.plot_loss_file_name)[:-4]+"_exp_"+str(i)+".png")
            os.rename(self.gradient_beta_norm_file_name, str(self.gradient_beta_norm_file_name)[:-5]+"_exp_"+str(i)+".json")
            # os.rename(self.plot_gradient_beta_norm_file_name, str(self.plot_gradient_beta_norm_file_name)[:-4]+"_exp_"+str(i)+".png")
            os.rename(self.gradient_alpha_clients_norm_file_name, str(self.gradient_alpha_clients_norm_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.gradient_personalized_objective_clients_norm_file_name, str(self.gradient_personalized_objective_clients_norm_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.gradient_delta_clients_norm_file_name, str(self.gradient_delta_clients_norm_file_name)[:-5]+"_exp_"+str(i)+".json")
            
        if (server_round + self.round_offset) % self.checkpoint_interval == 0:
            print(Fore.GREEN + f"{self.client_algorithm}: Saving aggregated weights at round {server_round + self.round_offset}" + Fore.WHITE)
            
            if self.dataset_name == 'stackoverflow_nwp':
                vocab_size = self.dataset_hyperparameters['vocab_size'] + 4
                embedding_size = self.dataset_hyperparameters['embedding_size']
                hidden_size = self.dataset_hyperparameters['hidden_size']
                num_layers = self.dataset_hyperparameters['num_layers']
                net = StackoverflowNet(vocab_size, embedding_size, hidden_size, num_layers)
            elif self.dataset_name == 'emnist':
                net = self.net

            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(
                {
                    'round': server_round + self.round_offset,
                    'net_state_dict': net.state_dict(),
                }, self.model_file_name
            )

        return aggregated_parameters_tuple

def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ):# -> Tuple[Optional[float], Dict[str, Scalar]]:

        # Weigh accuracy of each client by number of examples used
        before_accuracies = [r.metrics["before_accuracy"] * r.num_examples for _, r in results]
        after_accuracies = [r.metrics["after_accuracy"] * r.num_examples for _, r in results]
        if self.dataset_name == 'movielens':
            losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        
        # Aggregate and print custom metric
        before_accuracy_aggregated = sum(before_accuracies) / sum(examples)
        after_accuracy_aggregated = sum(after_accuracies) / sum(examples)
        print(Fore.RED + f"{self.client_algorithm}: Round {server_round + self.round_offset} accuracy aggregated from client results: {after_accuracy_aggregated}" + Fore.WHITE)
        self.before_accuracies_dict[int(server_round + self.round_offset)] = before_accuracy_aggregated
        self.after_accuracies_dict[int(server_round + self.round_offset)] = after_accuracy_aggregated
        
        all_before_accuracies = [r.metrics["before_accuracy"] for _, r in results]
        self.all_before_accuracies_dict[int(server_round + self.round_offset)] = all_before_accuracies
        all_after_accuracies = [r.metrics["after_accuracy"] for _, r in results]
        self.all_after_accuracies_dict[int(server_round + self.round_offset)] = all_after_accuracies

        if self.dataset_name == 'movielens':
            loss_aggregated = math.sqrt(sum(losses) / sum(examples))
            print(Fore.RED + f"{self.client_algorithm}: Round {server_round + self.round_offset} loss aggregated from client results: {loss_aggregated}" + Fore.WHITE)
            self.test_losses_dict[int(server_round + self.round_offset)] = loss_aggregated

        if (server_round + self.round_offset) % self.checkpoint_interval == 0:
            with open(self.before_accu_file_name, 'w') as f:
                json.dump(self.before_accuracies_dict, f)
            with open(self.after_accu_file_name, 'w') as f:
                json.dump(self.after_accuracies_dict, f)

            with open(self.all_before_accu_file_name, 'w') as f:
                json.dump(self.all_before_accuracies_dict, f)
            with open(self.all_after_accu_file_name, 'w') as f:
                json.dump(self.all_after_accuracies_dict, f)

            plt.figure(2)
            plt.plot(self.before_accuracies_dict.keys(), self.before_accuracies_dict.values(), c='#2978A0', label='Average')
            # plt.plot(*zip(*sorted(self.accuracies_dict.items())), c='#ed4f46')
            # mins = [min(vals) for vals in self.all_accuracies_dict.values()]
            # maxs = [max(vals) for vals in self.all_accuracies_dict.values()]
            # plt.fill_between(self.all_accuracies_dict.keys(), mins, maxs, color= '#fad3d1')
            # percentile25 = [sorted(vals)[2] for vals in self.all_accuracies_dict.values()]
            # percentile75 = [sorted(vals)[-3] for vals in self.all_accuracies_dict.values()]
            # plt.plot(self.all_accuracies_dict.keys(), percentile25, c='#694873', label='25th %ile')
            # plt.plot(self.all_accuracies_dict.keys(), percentile75, c='#F5F749', label='75th %ile')
            # plt.legend()
            plt.xlabel('Number of Rounds')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracy for ' + str(self.client_algorithm) + ' Baseline')
            plt.savefig(self.plot_before_accu_file_name)

            plt.figure(3)
            plt.plot(self.after_accuracies_dict.keys(), self.after_accuracies_dict.values(), c='#2978A0', label='Average')
            plt.xlabel('Number of Rounds')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracy for ' + str(self.client_algorithm) + ' Baseline')
            plt.savefig(self.plot_after_accu_file_name)

            if self.dataset_name == 'movielens':
                with open(self.test_loss_file_name, 'w') as f:
                    json.dump(self.test_losses_dict, f)
                plt.figure(4)
                # plt.plot(*zip(*sorted(self.accuracies_dict.items())), c='#ed4f46')
                plt.plot(self.test_losses_dict.keys(), self.test_losses_dict.values(), c='#ed4f46')
                plt.xlabel('Number of Rounds')
                plt.ylabel('Validation RMSE Loss')
                plt.title('Validation RMSE Loss for ' + str(self.client_algorithm) + ' Baseline')
                plt.savefig(self.plot_test_loss_file_name)

        if (server_round + self.round_offset) == self.total_rounds:
            i = 0
            while os.path.exists(self.after_accu_file_name[:-5]+"_exp_%s.json" %i):
                i += 1

            os.rename(self.before_accu_file_name, str(self.before_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.all_before_accu_file_name, str(self.all_before_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.after_accu_file_name, str(self.after_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.all_after_accu_file_name, str(self.all_after_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.plot_before_accu_file_name, str(self.plot_before_accu_file_name)[:-4]+"_exp_"+str(i)+".png")
            os.rename(self.plot_after_accu_file_name, str(self.plot_after_accu_file_name)[:-4]+"_exp_"+str(i)+".png")
            os.rename(self.model_file_name, str(self.model_file_name)[:-4]+"_exp_"+str(i)+".pth")
            
            if self.dataset_name == 'movielens':
                os.rename(self.test_loss_file_name, str(self.test_loss_file_name)[:-5]+"_exp_"+str(i)+".json")
                os.rename(self.plot_test_loss_file_name, str(self.plot_test_loss_file_name)[:-4]+"_exp_"+str(i)+".png")

            os.rename(self.personalized_plot_accu_file_name, str(self.personalized_plot_accu_file_name)[:-4]+"_exp_"+str(i)+".png")
            os.rename(self.personalized_accu_file_name, str(self.personalized_accu_file_name)[:-5]+"_exp_"+str(i)+".json")
            os.rename(self.personalized_all_accu_file_name, str(self.personalized_all_accu_file_name)[:-5]+"_exp_"+str(i)+".json")

        return super().aggregate_evaluate(server_round, results, failures)
    
class ScaledAttention(torch.nn.Module):
    
    def __init__(self, temperature, dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)
        
    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(-1, -2))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.w_qs = nn.Linear(n_head * d_k, n_head*d_model, bias=False)
        self.w_ks = nn.Linear(n_head * d_k, n_head*d_model, bias=False)
        self.w_vs = nn.Linear(n_head * d_v, n_head*d_model, bias=False)
        
#         nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(1.0 / (d_k)))
#         nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(1.0 / (d_k)))
#         nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(1.0 / (d_v)))
        
        self.attention = ScaledAttention(temperature=np.power(d_k, 0.5), dropout=dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v)
        
        self.fc = nn.Linear(n_head * d_model, n_head * d_v)
        
        # nn.init.xavier_normal_(self.fc.weight)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, fedAvg=None, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        B, N_src, _ = q.size()
        B, N_ngh, _ = k.size()
        B, N_ngh, _ = v.size()
        assert(N_ngh % N_src == 0)
        num_neighbors = int(N_ngh / N_src)
        
        residual = q
        q = self.w_qs(q).view(B, N_src, 1, n_head, self.d_model)

        k = self.w_ks(k).view(B, N_src, num_neighbors, n_head, self.d_model)
        v = self.w_vs(v).view(B, N_src, num_neighbors, n_head, self.d_model)
        
        q = q.transpose(2, 3).contiguous().view(B*N_src*n_head, 1, self.d_model)
        k = k.transpose(2, 3).contiguous().view(B*N_src*n_head, num_neighbors, self.d_model)
        v = v.transpose(2, 3).contiguous().view(B*N_src*n_head, num_neighbors, self.d_model)
        
        output, attn_map = self.attention(q, k, v, mask=mask)
        output = output.view(B, N_src, n_head*self.d_model)
        output = self.dropout(self.fc(output))
        if fedAvg is not None:
            output = self.layer_norm(output + fedAvg)
        else:
            output = self.layer_norm(output)
        attn_map = attn_map.view(B, N_src, n_head, num_neighbors)
        return output, attn_map

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        
        # torch.nn.init.xavier_normal_(self.fc1.weight)
        # torch.nn.init.xavier_normal_(self.fc2.weight)
    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        h = self.act(self.fc1(x))
        z = self.fc2(h)
        return z
    
    
class AttnModel(torch.nn.Module):
    def __init__(self, embedding_dim, model_dim,
                 n_head=1, drop_out=0.0):
        super(AttnModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        
        # self.merger = MergeLayer(self.model_dim, model_dim, model_dim, model_dim)
        
        self.multi_head_target = MultiHeadAttention(n_head,
                                                    d_model=32,
                                                    d_k = self.embedding_dim // n_head,
                                                    d_v = self.model_dim // n_head,
                                                    dropout = drop_out)
        
    def forward(self, src, seq, value, fedAvg=None):
        batch, N_src, _ = src.shape
        N_ngh = seq.shape[1]
        
        device = src.device
        output, attn = self.multi_head_target(src, seq, value, fedAvg)
        # output = self.merger(output, src)
        return output, attn