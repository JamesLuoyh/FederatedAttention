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

from colorama import Fore
from typing import Optional, List, OrderedDict, Tuple, Dict
from flwr.common import Parameters, Scalar, MetricsAggregationFn, FitIns, EvaluateIns, FitRes, EvaluateRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from models.stackoverflow_net import StackoverflowNet

from configs.hyperparameters import *

class FedAvg(fl.server.strategy.FedAvg):

    def __init__(
        self,
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

        super().__init__(fraction_fit, fraction_eval, min_fit_clients, min_eval_clients, min_available_clients, initial_parameters=initial_parameters, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ):# -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        client_proxy_and_fitins = super().configure_fit(rnd, parameters, client_manager)
        for i in client_proxy_and_fitins:
            i[1].config['rnd'] = rnd
        
        return client_proxy_and_fitins

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ):# -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        client_proxy_and_fitins = super().configure_evaluate(rnd, parameters, client_manager)
        for i in client_proxy_and_fitins:
            i[1].config['rnd'] = rnd

        # if self.client_algorithm == 'APFL' or self.client_algorithm == 'PerFedAvg':
        with open(self.checkpoint_dir + 'intermediate_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '_test_accuracies.txt') as f:
            accuracies = f.readlines()
            accuracies = [float(acc.strip()) for acc in accuracies]
            
            accuracy_aggregated = sum(accuracies) / len(accuracies)
            print(Fore.RED + f"{self.client_algorithm}: Round {rnd + self.round_offset} personalized accuracy aggregated from client results: {accuracy_aggregated}" + Fore.WHITE)
            os.remove(self.checkpoint_dir + 'intermediate_' + self.client_algorithm + '_' + str(self.num_clients) + '_' + str(self.total_clients) + '_' + str(self.total_rounds) + '_' + str(self.epochs) + '_' + str(self.lr).replace('.', '_') + '_test_accuracies.txt')
            
            self.personalized_accuracies_dict[int(rnd + self.round_offset)] = accuracy_aggregated
            self.all_personalized_accuracies_dict[int(rnd + self.round_offset)] = accuracies

        if (rnd + self.round_offset) % self.checkpoint_interval == 0:
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
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ):# -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        loss_aggregated = sum(losses) / sum(examples)
        print("")
        print(Fore.BLUE + f"{self.client_algorithm}: Round {rnd + self.round_offset} loss aggregated from client results: {loss_aggregated}" + Fore.WHITE)

        self.losses_dict[int(rnd + self.round_offset)] = loss_aggregated

        all_losses = [r.metrics["loss"] for _, r in results]
        self.all_losses_dict[int(rnd + self.round_offset)] = all_losses

        all_gradient_alpha_clients = [r.metrics['gradient_alpha'] for _, r in results]
        self.gradient_alpha_clients_norms_dict[int(rnd + self.round_offset)] = all_gradient_alpha_clients

        all_gradient_personalized_objective_clients = [r.metrics['gradient_personalized_objective'] for _, r in results]
        self.gradient_personalized_objective_clients_norms_dict[int(rnd + self.round_offset)] = all_gradient_personalized_objective_clients
        
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple
        aggregated_weights = fl.common.parameters_to_weights(aggregated_parameters)

        all_delta = []
        for _, r in results:
            weights = fl.common.parameters_to_weights(r.parameters)
            grad_norm = 0.0
            for client_weight, aggregated_weight in zip(weights, aggregated_weights):
                grad_norm += np.linalg.norm(client_weight - aggregated_weight) ** 2
            grad_norm = grad_norm ** (1. / 2)
            all_delta.append(grad_norm)
        self.gradient_delta_clients_norms_dict[int(rnd + self.round_offset)] = all_delta

        if (rnd + self.round_offset) % self.checkpoint_interval == 0:
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

        if (rnd + self.round_offset) == self.total_rounds:
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
            
        if (rnd + self.round_offset) % self.checkpoint_interval == 0:
            print(Fore.GREEN + f"{self.client_algorithm}: Saving aggregated weights at round {rnd + self.round_offset}" + Fore.WHITE)
            
            if self.dataset_name == 'stackoverflow_nwp':
                vocab_size = self.dataset_hyperparameters['vocab_size'] + 4
                embedding_size = self.dataset_hyperparameters['embedding_size']
                hidden_size = self.dataset_hyperparameters['hidden_size']
                num_layers = self.dataset_hyperparameters['num_layers']
                net = StackoverflowNet(vocab_size, embedding_size, hidden_size, num_layers)

            params_dict = zip(net.state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(
                {
                    'round': rnd + self.round_offset,
                    'net_state_dict': net.state_dict(),
                }, self.model_file_name
            )

        return aggregated_parameters_tuple

    def aggregate_evaluate(
        self,
        rnd: int,
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
        print(Fore.RED + f"{self.client_algorithm}: Round {rnd + self.round_offset} accuracy aggregated from client results: {after_accuracy_aggregated}" + Fore.WHITE)
        self.before_accuracies_dict[int(rnd + self.round_offset)] = before_accuracy_aggregated
        self.after_accuracies_dict[int(rnd + self.round_offset)] = after_accuracy_aggregated
        
        all_before_accuracies = [r.metrics["before_accuracy"] for _, r in results]
        self.all_before_accuracies_dict[int(rnd + self.round_offset)] = all_before_accuracies
        all_after_accuracies = [r.metrics["after_accuracy"] for _, r in results]
        self.all_after_accuracies_dict[int(rnd + self.round_offset)] = all_after_accuracies

        if self.dataset_name == 'movielens':
            loss_aggregated = math.sqrt(sum(losses) / sum(examples))
            print(Fore.RED + f"{self.client_algorithm}: Round {rnd + self.round_offset} loss aggregated from client results: {loss_aggregated}" + Fore.WHITE)
            self.test_losses_dict[int(rnd + self.round_offset)] = loss_aggregated

        if (rnd + self.round_offset) % self.checkpoint_interval == 0:
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

        if (rnd + self.round_offset) == self.total_rounds:
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

        return super().aggregate_evaluate(rnd, results, failures)