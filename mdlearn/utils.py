import os
import numpy as np
from . import fitting, preprocessing, encoding


def ml_predict(dir, smiles_list, t_list, p_list, encoders=None):
    '''
    This function is mainly called by other scripts
    :param dir:
    :param smiles_list:
    :param t_list:
    :param p_list:
    :return:
    '''

    model = fitting.TorchMLPRegressor(None, None, [])
    model.is_gpu = False
    model.load(os.path.join(dir, 'model.pt'))

    scaler = preprocessing.Scaler()
    scaler.load(os.path.join(dir, 'scale.txt'))

    if encoders is None:
        encoders = ['predefinedmorgan1', 'simple']
    encoder = encoding.FPEncoder(encoders, fp_name=os.path.join(dir, 'fp'))

    args = [np.array(l) for l in [smiles_list, t_list, p_list] if len(l) != 0]
    encoder.load_data(*args)
    datax = encoder.encode(save_fp=False, silent=True)
    datax = scaler.transform(datax)
    datay = model.predict_batch(datax)

    return datay
