import numpy as np
import pickle

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, s, a, r, s_next, done):

        data = (s, a, r, s_next, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx+1) % self._maxsize


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
        batch = np.random.choice(range(len(self._storage)), batch_size, replace=False)
        s, a, r, s_next, done = [], [], [], [], []
        for i in batch:
            s_temp, a_temp, r_temp, sn_temp, done_temp = self._storage[i]
            s.append(s_temp)
            a.append(a_temp)
            r.append(r_temp)
            s_next.append(sn_temp)
            done.append(done_temp)

        return (np.array(s),
                np.array(a),
                np.array(r),
                np.array(s_next),
                np.array(done))
