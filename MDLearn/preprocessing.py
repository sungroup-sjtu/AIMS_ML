""" Preprocessing
"""

import sklearn.model_selection
import sklearn.preprocessing
import numpy as np 

def random_select(*arrays, **options):
    """ Random choosing a fraction of arrays.
    Kwargs:
        One of 'size' and 'frac' must be present.
        'size': int (number of selection)/float (portion of selection);
        'frac': portion of selection;
    """
    for a in arrays[1:]:
        assert len(a) == len(arrays[0])

    size = options.get('size', None)
    if isinstance(size, float):
        size = int(size * len(arrays[0]))

    if not size:
        frac = options.get('frac', None)
        size = int(frac * len(arrays[0]))
    
    assert size > 0 and size <= len(arrays[0])

    # TODO Not correct. np.random.choice produce duplicated selections
    selector = np.zeros(len(arrays[0]), dtype=bool)
    for i in np.random.choice(len(arrays[0]), size):
        selector[i] = True 

    return [a[selector] for a in arrays] if len(arrays) > 1 else arrays[0][selector]


def separate(*arrays, **options):
    """ Random dividing arrays into two parts.
    Kwargs:
        'size': int (number of selection)/float (portion of selection)
    """

    if not 'train_size' in options:
        options['train_size'] = options.pop('size')

    return sklearn.model_selection.train_test_split(*arrays, **options)


class Selector:
    """ OOP Selector
    """

    def __init__(self, *data):
        """ data: np.array/pandas.df
        """
        self.training_index = None
        self.validation_index = None
        self.test_index = None

        for d in data[1:]:
            assert len(d) == len(data[0]), 'Data does not have same length'

        self.data = data
        self.selected_num = np.arange(len(self.data[0]))
        self.post_select_index = np.ones(len(self.data[0]), dtype=bool)

    def partition(self, training_size, validation_size=None):
        """ Perform 2-partition (training + testing) or 3-partition (training + validating + testing)
        """

        self.training_index = np.zeros(len(self.data[0]), dtype=bool)
        self.test_index = np.zeros(len(self.data[0]), dtype=bool)

        training_num, test_num = separate(self.selected_num, size=training_size)

        self.training_index[training_num] = True

        if not validation_size:
            self.test_index[test_num] = True
            self.validation_index = None

        else:
            validation_num, test_num = separate(test_num, size=validation_size/(1-training_size))
            self.validation_index = np.zeros(len(self.data[0]), dtype=bool)
            self.validation_index[validation_num] = True
            self.test_index[test_num] = True 

    def training_set(self):
        idx = np.logical_and(self.training_index, self.post_select_index)
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def validation_set(self):
        idx = np.logical_and(self.validation_index, self.post_select_index)        
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def test_set(self):
        idx = np.logical_and(self.test_index, self.post_select_index)                
        return (d[idx] for d in self.data) if len(self.data) > 1 else self.data[0][idx]

    def load(self, filename):

        with open(filename, 'r') as f_cache:
            line1 = f_cache.readline().rstrip('\n')
            line2 = f_cache.readline().rstrip('\n')
            line3 = f_cache.readline().rstrip('\n')

            self.training_index = np.array(list(map(int, line1)), dtype=bool)
            self.test_index = np.array(list(map(int, line2)), dtype=bool)

            if line3:
                self.validation_index = np.array(list(map(int, line3)), dtype=bool)

    def save(self, filename):

        with open(filename, 'w') as f_cache:
            f_cache.write(''.join(map(str, self.training_index.astype(int))))
            f_cache.write('\n')
            f_cache.write(''.join(map(str, self.test_index.astype(int))))
            f_cache.write('\n')

            if self.validation_index is not None:
                f_cache.write(''.join(map(str, self.validation_index.astype(int))))
                f_cache.write('\n')

        
class Scaler:
    """ StandardScaler that can save
    """

    def __init__(self, *args, **kwargs):
        self.scaler_ = sklearn.preprocessing.StandardScaler(*args, **kwargs)

    def fit(self, *args):
        return self.scaler_.fit(*args)

    def transform(self, *args):
        return self.scaler_.transform(*args)

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(','.join(map(str, self.scaler_.mean_)) + '\n')
            f.write(','.join(map(str, self.scaler_.var_)) + '\n')
            f.write(','.join(map(str, self.scaler_.scale_)) + '\n')

    def load(self, filename):
        self.scaler_.scale_ = 1
        with open(filename, 'r') as f:
            line1 = f.readline().strip('\n')
            line2 = f.readline().strip('\n')
            line3 = f.readline().strip('\n')

            self.scaler_.mean_ = np.array(list(map(float, line1.split(','))))
            self.scaler_.var_ = np.array(list(map(float, line2.split(','))))
            self.scaler_.scale_ = np.array(list(map(float, line3.split(','))))
            assert len(self.scaler_.mean_) == len(self.scaler_.var_)
