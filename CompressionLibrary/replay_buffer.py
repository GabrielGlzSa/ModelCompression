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


class PrioritizedReplayBufferMultipleDatasets(object):
    def __init__(self, size:int, alpha:int, dataset_names:list):
        
        self.dataset_names = dataset_names

        self._maxsize = size
        self.alpha = alpha

        # Dictionary to get the index of each dataset.
        self.dataset_dict = dict(zip(dataset_names,range(len(self.dataset_names))))

        # Create variables to store data.
        self.reset()


    def __len__(self):
        return np.sum(list(map(lambda x: len(self._states[x]), self.dataset_names)))
        
    def reset(self):
        self._states = dict(zip(self.dataset_names, map(lambda x: [], self.dataset_names)))
        self._next_states = dict(zip(self.dataset_names, map(lambda x: [], self.dataset_names)))

        num_datasets = len(self.dataset_names)
        self._actions = np.zeros((num_datasets, self._maxsize), dtype=np.int32)
        self._rewards = np.zeros((num_datasets, self._maxsize), dtype=np.float32)
        self._done = np.full((num_datasets, self._maxsize), dtype=np.bool8)

        self._td_error = np.zeros((num_datasets, self._maxsize), dtype=np.float32)
        self._next_idx = np.zeros(num_datasets, dtype=np.int32)
        self._max_saved = np.zeros(num_datasets, dtype=np.int32)
        
    def add(self, s, a, rw, done, td_error, dataset_name):

        dataset_index = self.dataset_dict[dataset_name]
        
        
        self._states[dataset_name].append(s)
        self._actions[dataset_index,self._next_idx[dataset_index]] = a
        self._rewards[dataset_index,self._next_idx[dataset_index]] = rw
        self._done[dataset_index,self._next_idx[dataset_index]] = done
        self._td_error[dataset_index,self._next_idx[dataset_index]] = td_error

        self._next_idx[dataset_index] = (self._next_idx[dataset_index]+1) % self._maxsize
        self._max_saved[dataset_index] = max(self._max_saved[dataset_index], self._next_idx[dataset_index])

        self.last_sampled = None
 

    def update_td_error(self, td_errors):

        batch_start = 0
        for dataset_idx, indexes in enumerate(self.last_sampled):
            batch_size = len(indexes)
            self._td_error[dataset_idx, indexes] = td_errors[batch_start:batch_start+batch_size]
            batch_start = batch_size



    def sample(self, batch_size):
        # Empty list for states
        s = []
        actions = []
        rewards = []
        s_next = []
        done = []

        num_datasets = len(self.dataset_names)
        batch_counter = 0
        recommended_batch_size = batch_size//num_datasets

        self.last_sampled = []

        for dataset_name in self._states.keys():
            dataset_idx = self.dataset_dict[dataset_name]
            max_dataset = self._max_saved[dataset_idx]
            td_error = self._td_error[dataset_idx, :max_dataset]
            td_error_sorted_idx = np.argsort(td_error)

            # Add the samples from the dataset that were used to train the network.
            self.last_sampled.append(td_error_sorted_idx)

            probabilities = np.empty_like(td_error)
            probabilities[td_error_sorted_idx] = list(range(1, max_dataset+1)) 
            probabilities = 1/probabilities
            # Use alpha to determine how much priorization.
            probabilities = np.power(probabilities, self.alpha)
            # Normalize to obtain probabilities
            probabilities = probabilities / np.sum(probabilities)

            if batch_counter + recommended_batch_size > batch_size:
                recommended_batch_size = batch_size - batch_counter

            if recommended_batch_size < len(self._states[dataset_name]):
                batch = np.random.choice(len(self._states[dataset_name]), recommended_batch_size, replace=False, p=probabilities)
                batch_counter += recommended_batch_size
            else:
                num_storage_sampes = len(self._states[dataset_name])
                batch = np.random.choice(len(self._states[dataset_name]), num_storage_sampes, replace=False, p=probabilities)
                batch_counter += num_storage_sampes


            for batch_element in batch:
                # Remove  dimensions of size 1 so that it can be stacked.
                s.append(tf.squeeze(self._states[dataset_name][batch_element]))
                s_next.append(tf.squeeze(self._states[dataset_name][batch_element]))

            dataset_index = self.dataset_dict[dataset_name]
            actions.extend(self._actions[dataset_index][batch])
            rewards.extend(self._rewards[dataset_index][batch])
            done.extend(self._done[dataset_index][batch])


        # Stack feature maps and add depth of 1.
        s = tf.expand_dims(tf.ragged.stack(s), axis=-1)
        s_next = tf.expand_dims(tf.ragged.stack(s_next), axis=-1)
        return (s.to_tensor(),
                tf.convert_to_tensor(actions, dtype=tf.int32),
                tf.convert_to_tensor(rewards, dtype=tf.float32),
                s_next.to_tensor(),
                tf.convert_to_tensor(done, dtype=tf.int32))