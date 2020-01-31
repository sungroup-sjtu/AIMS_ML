""" Higher-level API for fingerprint
"""

import pathlib
import numpy as np
from mdlearn.fp import encoders_dict


class FPEncoder:
    """ Handles higher level processing of fingerprints
    """

    def __init__(self, encoders, fp_name, save_svg=False):
        """ indexer_class: A class that has function index(object) ==> numpy array
        """
        self.Indexers = []
        self.IndexerPara = []
        for encoder in encoders:
            if len(encoder.split('-')) == 1:
                self.Indexers.append(encoders_dict.get(encoder))
                self.IndexerPara.append(None)
            else:
                self.Indexers.append(encoders_dict.get(encoder.split('-')[0]))
                self.IndexerPara.append(list(map(int, encoder.split('-')[1:])))
        if None in self.Indexers:
            raise Exception('Available encoders: %s' % encoders_dict.keys())

        self.fp_name = fp_name
        self.save_svg = save_svg

    def load_data(self, smiles_list, *other_lists):
        """ Other lists will be vstacked into final fingerprint
        """

        self.smiles_list = smiles_list[:]

        for ol in other_lists:
            assert ol.shape == smiles_list.shape

        self.other_lists = list(other_lists)

    def encode(self, save_fp=True, silent=False):
        ret_list = []

        for i, Indexer in enumerate(self.Indexers):
            if self.IndexerPara[i] is None:
                idxer = Indexer()
            else:
                idxer = Indexer(self.IndexerPara[i])
            idxer._silent = silent
            if idxer.use_pre_idx_list is not None:
                filename = self.fp_name + '_' + idxer.use_pre_idx_list + '-' + \
                           '-'.join(list(map(str, self.IndexerPara[i]))) + '.idx'
                with open(filename) as f:
                    idxer.pre_idx_list = f.read().splitlines()

            if self.save_svg:
                idxer.svg_dir = pathlib.Path(self.fp_name + '-svg_' + Indexer.name)

            results_list = idxer.index_list(self.smiles_list)

            fp_filename = self.fp_name + '_' + Indexer.name
            if hasattr(idxer, 'fp_time_limit'):
                fp_filename += '-' + str(idxer.fp_time_limit)
            if hasattr(idxer, 'maxPath'):
                fp_filename += '-' + str(idxer.maxPath)
            if save_fp:
                with open(fp_filename, 'w') as f_fp:
                    for name, result in zip(self.smiles_list, results_list):
                        f_fp.write('%s %s\n' % (name, ','.join(map(str, result))))

            if not idxer.use_pre_idx_list and len(idxer.idx_list) > 0:
                with open(fp_filename + '.idx', 'w') as f_idx:
                    for idx in idxer.idx_list:
                        f_idx.write(f'{idx}\n')

            ret_list.append(np.array(results_list))

        ret = np.hstack(ret_list)

        if self.other_lists == []:
            return ret
        else:
            return np.hstack([ret] + [s[:, np.newaxis] for s in self.other_lists])
