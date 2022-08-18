from abc import abstractmethod

import numpy as np
import torch
import time

from lib.meta_rl.algorithms.pacoh.modules.data_sampler import MetaDatasetSampler


class MetaLearner:
    plot_functions_during_training = False
    meta_train_config = None
    meta_test_config = None

    def __init__(self, random_seed=None):
        self._set_seed(random_seed)

    def _standardize(self, x, y=None):
        x = (x - self.x_mean) / self.x_std
        if y is not None:
            y = (y - self.y_mean) / self.y_std
            return x, y
        else:
            return x

    def _unstandardize(self, y):
        y = y * self.y_std + self.y_mean
        return y

    def _process_meta_train_data(self, meta_train_data, meta_batch_size, batch_size, n_batched_models_train):
        self.num_meta_train_tasks = len(meta_train_data)
        self.meta_train_sampler = MetaDatasetSampler(meta_train_data, batch_size, meta_batch_size=meta_batch_size,
                                                             n_batched_models=n_batched_models_train, tiled=True)
        self.x_mean, self.y_mean, self.x_std, self.y_std = self.meta_train_sampler.get_standardization_stats()
        self.input_dim = self.meta_train_sampler.input_dim
        self.output_dim = self.meta_train_sampler.output_dim

    @abstractmethod
    def step(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _predict(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _setup_meta_test_models_and_step(self, meta_test_tasks):
        raise NotImplementedError

    def _meta_test_training_loop(self, test_tasks, eval_models, eval_models_step,
                                 num_iter=3000, log_period=1000, eval_period=1000):
        t = time.time()
        for iter in range(num_iter):
            log_prob = eval_models_step()

            # print meta-test training loss
            if iter > 0 and iter % log_period == 0:
                avg_log_prob = log_prob.mean().numpy()
                message = '\tMeta-Test - Iter %d/%d - Time %.2f sec' % (iter, num_iter, time.time() - t)
                message += ' - Train-Log-Prob: %.3f' % avg_log_prob
                print(message)
                t = time.time()

            # meta-test evaluation
            if iter > 0 and iter % eval_period == 0:
                t = time.time()
                eval_metrics_mean, eval_metrics_std, _ = self._meta_test_models_eval(test_tasks, eval_models)
                message = ""
                for key in eval_metrics_mean:
                    message += '- Val-%s: %.3f +- %.3f' % (key, eval_metrics_mean[key], eval_metrics_std[key])

                print('\n\t Iter %d/%d - Time %.2f sec %s' % (iter, num_iter, time.time() - t, message))

    def _meta_test_models_eval(self, test_tasks, eval_models):
        eval_metrics_grouped = {}
        for task, model in zip(test_tasks, eval_models):
            res = model.eval(*task.val_data)
            for key in res:
                l = eval_metrics_grouped.get(key, [])
                l.append(res[key])
                eval_metrics_grouped[key] = l

        eval_metrics_mean = {}
        eval_metrics_std = {}
        for key in eval_metrics_grouped:
            eval_metrics_mean[key] = np.mean(eval_metrics_grouped[key])
            eval_metrics_std[key] = np.std(eval_metrics_grouped[key])

        return eval_metrics_mean, eval_metrics_std, eval_metrics_grouped

    def _set_seed(self, random_seed):
        if random_seed is not None:
            np.random.seed(random_seed + 1)
            torch.manual_seed(random_seed + 2)

    @staticmethod
    def _aggregate_eval_metrics_across_tasks(eval_metrics_dict_per_task):
        eval_metrics_mean, eval_metrics_std = {}, {}
        metric_keys = set.union(*[set(metrics_dict.keys()) for metrics_dict in eval_metrics_dict_per_task])
        for key in metric_keys:
            metric_list = []
            for eval_metrics_dict in eval_metrics_dict_per_task:
                if type(eval_metrics_dict[key]) == list:
                    metric_list.extend(eval_metrics_dict[key])
                else:
                    metric_list.append(eval_metrics_dict[key])
            eval_metrics_mean[key] = np.mean(metric_list)
            eval_metrics_std[key] = np.std(metric_list)
        return eval_metrics_mean, eval_metrics_std

    @staticmethod
    def _split_into_batches(list, max_batch_size):
        import math
        n_elements = len(list)
        if max_batch_size == -1:
            max_batch_size = n_elements
        n_batches = math.ceil(n_elements / float(max_batch_size))
        remainder = n_elements % n_batches
        batches = []
        idx = 0
        for i in range(n_batches):
            if i < remainder:
                batch_size = n_elements // n_batches + 1
            else:
                batch_size = n_elements // n_batches
            batches.append(list[idx:idx + batch_size])
            idx += batch_size

        assert all([len(batch) <= max_batch_size for batch in batches])
        assert np.sum([len(batch) for batch in batches]) == n_elements
        return batches
