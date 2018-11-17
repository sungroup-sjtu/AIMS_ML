""" Higher-level API for fingerprint
"""

import numpy as np
from multiprocessing import Pool

from functools import partial


class FPEncoder:
    """ Handles higher level processing of fingerprints
    """

    def __init__(self, indexer_class, selector=None, cache_filename=None, smiles_decoder='pybel'):
        """ indexer_class: A class that has function index(object) ==> numpy array
        """

        self.Indexer = indexer_class
        self.selector = selector
        self.cache_filename = cache_filename

        if smiles_decoder == 'pybel':
            import pybel
            self.smiles_decoder = _encode_smiles_pybel

        elif smiles_decoder == 'rdk':
            import rdkit.Chem
            self.smiles_decoder = rdkit.Chem.MolFromSmiles

        else:
            raise NotImplementedError()

    def load_data(self, smiles_list, *other_lists):
        """ Other lists will be vstacked into final fingerprint
        """

        self.smiles_list = smiles_list[:]

        for ol in other_lists:
            assert ol.shape == smiles_list.shape

        self.other_lists = list(other_lists)

        self.data_cache = dict()

        if self.cache_filename:
            try:
                self.load_file()
            except IOError:
                pass

    def load_file(self):

        with open(self.cache_filename, 'r') as f_cache:
            for line in f_cache:

                line = line.strip('\n')
                name, rd = line.split(' ')
                d = np.fromstring(rd, dtype=float, sep=',')

                assert name not in self.data_cache, name + ' repeat'
                self.data_cache[name] = d

    def encode(self, nproc=1):

        pending_queue = list(set(self.smiles_list).difference(self.data_cache.keys()))

        if nproc == 1 or len(pending_queue) < nproc:
            pending_results = _encode_list(pending_queue, self.smiles_decoder, self.Indexer())

        else:
            with Pool(nproc) as pool:

                pending_results = pool.map(
                    partial(_encode_list, smiles_decoder=self.smiles_decoder, index_encoder=self.Indexer()),
                    pending_queue)

        self.data_cache.update(dict(zip(pending_queue, pending_results)))

        if self.cache_filename:
            with open(self.cache_filename, 'a') as f_cache:
                for name, result in zip(pending_queue, pending_results):
                    f_cache.write('%s %s\n' % (name, ','.join(map(str, result))))

        if not self.selector:
            ret = np.array([self.data_cache[name] for name in self.smiles_list])
        else:
            ret = np.array([self.data_cache[name][self.selector] for name in self.smiles_list])

        if not self.other_lists:
            return ret
        else:
            return np.hstack([ret] + [s[:, np.newaxis] for s in self.other_lists])


def _encode_smiles_pybel(string):

    import pybel
    return pybel.readstring('smi', string)

def _encode_list(smiles_list, smiles_decoder, index_encoder):

    return [index_encoder.index(smiles_decoder(name)) for name in smiles_list]

