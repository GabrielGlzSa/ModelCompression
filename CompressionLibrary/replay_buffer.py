import numpy as np
import pickle
import tensorflow as tf

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

