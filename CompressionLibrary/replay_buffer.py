import numpy as np
import pickle
import tensorflow as tf
import logging

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def __str__(self) -> str:
        msg = ''
        for s, a, r, s_next, done in self._storage:
            msg+=f'{s.shape}, {a}, {r}, {s_next.shape}, {done}\n'

        return msg
        

    def add(self, s, a, r, s_next, done):

        data = (s, a, r, s_next, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx+1) % self._maxsize

    def add_multiple(self, s, a, r, s_next, done):
        batch_size = s.shape[0]
        for i in range(batch_size):
            self.add(s[i], a[i], r[i], s_next[i], done[i])

    def save(self, path):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self._storage, f)
        except KeyboardInterrupt:
            print('Saving file again as it was previously interrupted.')
            with open(path, 'wb') as f:
                pickle.dump(self._storage, f)
            print('Succesfuly saved.')

    def load(self, path):
        with open(path, 'rb') as f:
          examples = pickle.load(f)
        if self._maxsize-len(self._storage) < len(examples):
          batch = np.random.choice(range(len(examples)), self._maxsize-len(self._storage), replace=False)
          for idx in batch:
            self._storage.append(examples[idx])
            self._next_idx = (self._next_idx+1) % self._maxsize
        else:
          for data in examples:
            self._storage.append(data)
            self._next_idx = (self._next_idx+1) % self._maxsize

    def sample(self, batch_size):
        if batch_size < len(self._storage):
            batch = np.random.choice(range(len(self._storage)), batch_size, replace=False)
        else:
            batch = np.random.choice(range(len(self._storage)), len(self._storage), replace=False)

        s, a, r, s_next, done = [], [], [], [], []
        for i in batch:
            s_temp, a_temp, r_temp, sn_temp, done_temp = self._storage[i]
            s.append(s_temp)
            a.append(a_temp)
            r.append(r_temp)
            s_next.append(sn_temp)
            done.append(done_temp)

        return (tf.ragged.stack(s),
                tf.convert_to_tensor(a),
                tf.convert_to_tensor(r),
                tf.ragged.stack(s_next),
                np.array(done))

class ReplayBufferMultipleDatasets(object):
    def __init__(self, size, dataset_names):
        self._storage = dict(zip(dataset_names, map(lambda x: [], dataset_names)))
        self._maxsize = size
        self._next_idx = dict(zip(dataset_names, map(lambda x: 0, dataset_names)))
        self.dataset_names = dataset_names

    def __len__(self):
        return np.sum(list(map(lambda x: len(self._storage[x]), self.dataset_names)))

    def __str__(self) -> str:
        msg = ''
        for key, value in self._storage.items():
            msg += f'Dataset: {key}'+'-o-'*5 + '\n'
            for s, a, r, s_next, done in value:
                msg+=f'{s.shape}, {a}, {r}, {s_next.shape}, {done}\n'

        return msg
        

    def add(self, s, a, r, s_next, done, dataset_name):

        data = (s, a, r, s_next, done)
        if self._next_idx[dataset_name] >= len(self._storage[dataset_name]):
            self._storage[dataset_name].append(data)
        else:
            self._storage[dataset_name][self._next_idx[dataset_name]] = data

        self._next_idx[dataset_name] = (self._next_idx[dataset_name]+1) % (self._maxsize//2)

    def add_multiple(self, s, a, r, s_next, done, dataset_name):
        batch_size = s.shape[0]
        for i in range(batch_size):
            self.add(s[i], a[i], r[i], s_next[i], done[i], dataset_name)

    def save(self, path):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self._storage, f)
        except KeyboardInterrupt:
            print('Saving file again as it was previously interrupted.')
            with open(path, 'wb') as f:
                pickle.dump(self._storage, f)
            print('Succesfuly saved.')

    def load(self, path):
        with open(path, 'rb') as f:
            examples = pickle.load(f)
            if self._maxsize-len(self._storage) < len(examples):
                batch = np.random.choice(range(len(examples)), self._maxsize-len(self._storage), replace=False)
                for idx in batch:
                    self._storage.append(examples[idx])
                    self._next_idx = (self._next_idx+1) % self._maxsize
            else:
                for data in examples:
                    self._storage.append(data)
                    self._next_idx = (self._next_idx+1) % self._maxsize

    def sample(self, batch_size):
        s, a, r, s_next, done = [], [], [], [], []

        num_datasets = len(self.dataset_names)
        batch_counter = 0
        recommended_batch_size = batch_size//num_datasets
        for dataset_name in self._storage.keys():

            if batch_counter + recommended_batch_size > batch_size:
                recommended_batch_size = batch_size - batch_counter

            if recommended_batch_size < len(self._storage[dataset_name]):
                batch = np.random.choice(range(len(self._storage[dataset_name])), recommended_batch_size, replace=False)
                batch_counter += recommended_batch_size
            else:
                num_storage_sampes = len(self._storage[dataset_name])
                batch = np.random.choice(range(len(self._storage[dataset_name])), num_storage_sampes, replace=False)
                batch_counter += num_storage_sampes

                
            for i in batch:
                s_temp, a_temp, r_temp, sn_temp, done_temp = self._storage[dataset_name][i]
                s.append(s_temp)
                a.append(a_temp)
                r.append(r_temp)
                s_next.append(sn_temp)
                done.append(done_temp)

        return (tf.ragged.stack(s),
                tf.convert_to_tensor(a),
                tf.convert_to_tensor(r),
                tf.ragged.stack(s_next),
                np.array(done))


class PrioritizedExperienceReplayBufferMultipleDatasets(object):
    def __init__(self, size, alpha=1):
        
        self.dataset_names = sorted(dataset_names)
        self.num_datasets = len(self.dataset_names)
        self._storage = dict(zip(self.dataset_names, map(lambda x: [], self.dataset_names)))
        self._td_error =  np.zeros((self.num_datasets, size), dtype=np.float32)
        self._next_idx = np.zeros(self.num_datasets, dtype=np.uint32)
        self._times_selected = np.zeros((self.num_datasets, size), dtype=np.uint32)
        self.alpha = alpha
        self._maxsize = size
        self.logger = logging.getLogger(__name__)
        self.idx_dict = dict(zip(self.dataset_names, range(self.num_datasets)))

    def __len__(self):
        return np.sum(list(map(lambda x: len(self._storage[x]), self.num_datasets)))

    def __str__(self) -> str:
        msg = ''
        for key, value in self._storage.items():
            msg += f'Dataset: {key}'+'-o-'*5 + '\n'
            for s, a, r, s_next, done in value:
                msg+=f'{s.shape}, {a}, {r}, {s_next.shape}, {done}\n'

        return msg
        
    def add(self, s, a, r, s_next, td_error, done, dataset_name):
        data_idx = self.idx_dict[dataset_name]
        data = (s, a, r, s_next, done)
        if self._next_idx[data_idx] >= len(self._storage[dataset_name]):
            self._storage[dataset_name].append(data)
        else:
            self._storage[dataset_name][self._next_idx[data_idx]] = data
        self._td_error[data_idx, self._next_idx[data_idx]] = td_error
        self._next_idx[data_idx] = (self._next_idx[data_idx]+1) % self._maxsize

    def add_multiple(self, s, a, r, s_next, td_error, done, dataset_name):
        batch_size = s.shape[0]
        for i in range(batch_size):
            self.add(s[i], a[i], r[i], s_next[i], td_error[i], done[i], dataset_name)

    def update_td_error(self, mixed_indexes, td_errors):
        for d_idx, indexes in enumerate(mixed_indexes):
            self._times_selected[d_idx, indexes] =  self._times_selected[d_idx, indexes] + 1
            # self.logger.debug(f'TD errors before: {self._td_error[indexes]}')
            td_error_before = np.mean(self._td_error[d_idx, indexes])
            self._td_error[d_idx, indexes] = td_errors
            # self.logger.debug(f'TD errors after: {self._td_error[indexes]}')
            self.logger.debug(f'Mean TD error before: {td_error_before}')
            self.logger.debug(f'Mean TD error after: {np.mean(self._td_error[d_idx, indexes])}')

    def sample(self, batch_size):
        length_storage = len(self._storage)
        self.logger.debug(f'Next index are {self._next_idx}.')
        self.logger.debug(f'Replay buffer size is {length_storage}.')
        if self._next_idx < length_storage:
            td_error = self._td_error[:]
            self.logger.debug(f'TD error lenght is {len(td_error)}.')
            td_error_sorted_idx = np.argsort(td_error)
            td_error_sorted_idx = np.flip(td_error_sorted_idx)
            probabilities = np.empty_like(td_error_sorted_idx, dtype=np.float32)
            probabilities[td_error_sorted_idx] = list(range(1, length_storage+1)) 
            probabilities = 1 / np.power(probabilities, self.alpha)
            probabilities = probabilities / np.sum(probabilities)
            if batch_size < length_storage:
                batch = np.random.choice(length_storage, batch_size, p=probabilities, replace=False)
            else:
                batch = np.random.choice(length_storage, length_storage, p=probabilities, replace=False)
        else:
            td_error = self._td_error[:self._next_idx]
            length_storage = self._next_idx
            self.logger.debug(f'TD error lenght is {len(td_error)}.')
            td_error_sorted_idx = np.argsort(td_error)
            td_error_sorted_idx = np.flip(td_error_sorted_idx)
            probabilities = np.empty_like(td_error_sorted_idx, dtype=np.float32)
            probabilities[td_error_sorted_idx] = list(range(1, length_storage+1)) 
            probabilities = 1 / np.power(probabilities, self.alpha)
            probabilities = probabilities / np.sum(probabilities)
            if batch_size < length_storage:
                batch = np.random.choice(length_storage, batch_size, p=probabilities, replace=False)
            else:
                batch = np.random.choice(length_storage, length_storage, p=probabilities, replace=False)

        
        s, a, r, s_next, done = [], [], [], [], []
        for i in batch:
            s_temp, a_temp, r_temp, sn_temp, done_temp = self._storage[i]
            s.append(s_temp)
            a.append(a_temp)
            r.append(r_temp)
            s_next.append(sn_temp)
            done.append(done_temp)

        probs = probabilities[batch]
        done = np.asarray(done, dtype=np.float32)
        
        return (tf.ragged.stack(s),
                tf.convert_to_tensor(a),
                tf.convert_to_tensor(r),
                tf.ragged.stack(s_next),
                done,
                tf.convert_to_tensor(probs),
                batch)


class PrioritizedExperienceReplayBuffer(object):
    def __init__(self, size, alpha=1):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.alpha = alpha
        self._td_error = np.zeros(size, dtype=np.float32)
        self._times_selected = np.zeros(size, dtype=np.uint32)
        self.logger = logging.getLogger(__name__)

    def __len__(self):
        return len(self._storage)

    def __str__(self) -> str:
        msg = ''
        for s, a, r, s_next, done in self._storage:
            msg+=f'{s.shape}, {a}, {r}, {s_next.shape}, {done}\n'

        return msg
        

    def add(self, s, a, r, s_next, td_error, done):

        data = (s, a, r, s_next, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._td_error[self._next_idx] = td_error
        self._next_idx = (self._next_idx+1) % self._maxsize

    def add_multiple(self, s, a, r, s_next, td_error, done):
        batch_size = s.shape[0]
        
        for i in range(batch_size):
            self.add(s[i], a[i], r[i], s_next[i], td_error[i], done[i])

    def update_td_error(self, indexes, td_errors):
        self._times_selected[indexes] =  self._times_selected[indexes] + 1
        self.logger.debug(f'TD errors before: {self._td_error[indexes]}')
        td_error_before = np.mean(self._td_error[indexes])
        self._td_error[indexes] = td_errors
        self.logger.debug(f'TD errors after: {self._td_error[indexes]}')
        self.logger.debug(f'Mean TD error before: {td_error_before}')
        self.logger.debug(f'Mean TD error after: {np.mean(self._td_error[indexes])}')

    def sample(self, batch_size, highest_td_error=True):
        length_storage = len(self._storage)
        self.logger.debug(f'Next index is {self._next_idx}.')
        self.logger.debug(f'Replay buffer size is {length_storage}.')
        if self._next_idx < length_storage:
            td_error = self._td_error[:]
            self.logger.debug(f'TD error lenght is {len(td_error)}.')
            td_error_sorted_idx = np.argsort(td_error)
            if highest_td_error:
                td_error_sorted_idx = np.flip(td_error_sorted_idx)

            probabilities = np.empty_like(td_error_sorted_idx, dtype=np.float32)
            probabilities[td_error_sorted_idx] = list(range(1, length_storage+1)) 
            probabilities = 1 / np.power(probabilities, self.alpha)
            probabilities = probabilities / np.sum(probabilities)
            if batch_size < length_storage:
                batch = np.random.choice(length_storage, batch_size, p=probabilities, replace=False)
            else:
                batch = np.random.choice(length_storage, length_storage, p=probabilities, replace=False)
        else:
            td_error = self._td_error[:self._next_idx]
            length_storage = self._next_idx
            self.logger.debug(f'TD error lenght is {len(td_error)}.')
            td_error_sorted_idx = np.argsort(td_error)
            td_error_sorted_idx = np.flip(td_error_sorted_idx)
            probabilities = np.empty_like(td_error_sorted_idx, dtype=np.float32)
            probabilities[td_error_sorted_idx] = list(range(1, length_storage+1)) 
            probabilities = 1 / np.power(probabilities, self.alpha)
            probabilities = probabilities / np.sum(probabilities)
            if batch_size < length_storage:
                batch = np.random.choice(length_storage, batch_size, p=probabilities, replace=False)
            else:
                batch = np.random.choice(length_storage, length_storage, p=probabilities, replace=False)

        
        s, a, r, s_next, done = [], [], [], [], []
        for i in batch:
            s_temp, a_temp, r_temp, sn_temp, done_temp = self._storage[i]
            s.append(s_temp)
            a.append(a_temp)
            r.append(r_temp)
            s_next.append(sn_temp)
            done.append(done_temp)

        probs = probabilities[batch]
        done = np.asarray(done, dtype=np.float32)
        
        return (tf.ragged.stack(s),
                tf.convert_to_tensor(a),
                tf.convert_to_tensor(r),
                tf.ragged.stack(s_next),
                done,
                tf.convert_to_tensor(probs),
                batch)

