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

        self.Indexers = [encoders_dict.get(encoder) for encoder in encoders]
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

    def encode(self):
        ret_list = []

        for Indexer in self.Indexers:
            idxer = Indexer()
            if self.save_svg:
                svg_dir = pathlib.Path(self.fp_name + '-svg_' + Indexer.name)
                if not svg_dir.exists():
                    svg_dir.mkdir()
                idxer.svg_dir = svg_dir

            results_list = idxer.index_list(self.smiles_list)

            fp_filename = self.fp_name + '_' + Indexer.name
            with open(fp_filename, 'w') as f_fp:
                for name, result in zip(self.smiles_list, results_list):
                    f_fp.write('%s %s\n' % (name, ','.join(map(str, result))))

            ret_list.append(np.array(results_list))

        ret = np.hstack(ret_list)

        if self.other_lists == []:
            return ret
        else:
            return np.hstack([ret] + [s[:, np.newaxis] for s in self.other_lists])