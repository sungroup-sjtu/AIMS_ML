from .. import encoding, preprocessing, fp

import pandas as pd
import numpy as np
import os.path

EXAMPLE_PATH = os.path.dirname(__file__)


def load_altp(filename=None, target='Density', load_names=False, featrm: [] = None, encoder='alkane'):
    """ Load alkane data from alkanes with different temperature and pressure;
        target: 'Ei'/'Density'/'Compressibility'/'Expansion'/'Cp'
        load_names: True to return name of each molecule additionally
        transform: Only load selected fp into the training set.
    """

    if filename is None or filename == '_example_':
        filename = os.path.join(EXAMPLE_PATH, 'altp/result-npt-Ane.txt')

    if encoder == 'alkane':
        indexer = fp.AlkaneSKIndexer
    elif encoder == 'ch':
        indexer = fp.CHSKIndexer
    else:
        raise Exception('Unknown fingerprint encoder:', encoder)

    encoder = encoding.FPEncoder(indexer,
                                 selector=None,
                                 cache_filename=filename + '.fpcache',
                                 smiles_decoder='pybel')
    df = pd.read_csv(filename, sep=' ', index_col=False)
    encoder.load_data(df['SMILES'], df['T'], df['P'])
    datax = encoder.encode()

    if featrm is not None and len(featrm) > 0:
        selector = np.ones(datax.shape[1], dtype=bool)
        selector[featrm] = False
        datax = datax[:, selector]

    if not load_names:
        return datax, df[target].as_matrix()[:, np.newaxis]
    else:
        names = []
        for name, t, p in zip(df['SMILES'], df['T'], df['P']):
            names.append('%s\t%.2e\t%.2e' % (name, t, p))

        return datax, df[target].as_matrix()[:, np.newaxis], np.array(names)
