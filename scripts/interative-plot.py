import sys
import argparse
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
sys.path.append('../mdlearn')
# import fitting, metrics, preprocessing, validation, dataloader
import fitting,preprocessing, dataloader
from bokeh.models.widgets import PreText, Select
from bokeh.layouts import widgetbox
from bokeh.models import Range1d,Label
from bokeh.layouts import row
from bokeh.models import ColumnDataSource
from bokeh.models import LassoSelectTool
from bokeh.plotting import figure, curdoc
from ipywidgets import interact
import numpy as np
from bokeh.io import show

from bokeh.plotting import figure, show, output_notebook, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import tools

import re
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import PandasTools
# PandasTools ~ rdDepictor
from rdkit import DataStructs
import re
def main():
    parser = argparse.ArgumentParser(description='Alkane property fitting demo')
    parser.add_argument('-i', '--input', type=str, help='Data')
    parser.add_argument('-f', '--fp', type=str, help='Fingerprints')
    parser.add_argument('-o', '--output', default='out', type=str, help='Output directory')
    parser.add_argument('-t', '--target', default='raw_density', type=str, help='Fitting target')
    parser.add_argument('-p', '--part', default='', type=str, help='Partition cache file')
    parser.add_argument('-l', '--layer', default='16,16', type=str, help='Size of hidden layers')
    parser.add_argument('--visual', default=1, type=int, help='Visualzation data')
    parser.add_argument('--gpu', default=1, type=int, help='Using gpu')
    parser.add_argument('--epoch', default="200", type=str, help='Number of epochs')
    parser.add_argument('--step', default=500, type=int, help='Number of steps trained for each batch')
    parser.add_argument('--batch', default=int(1e9), type=int, help='Batch size')
    parser.add_argument('--lr', default="0.005", type=str, help='Initial learning rate')
    parser.add_argument('--l2', default=0.000, type=float, help='L2 Penalty')
    parser.add_argument('--check', default=10, type=int,
                        help='Number of epoch that do convergence check. Set 0 to disable.')
    parser.add_argument('--minstop', default=0.2, type=float, help='Minimum fraction of step to stop')
    parser.add_argument('--maxconv', default=2, type=int, help='Times of true convergence that makes a stop')
    parser.add_argument('--featrm', default='', type=str, help='Remove features')
    parser.add_argument('--optim', default='rms', type=str, help='optimizer')
    parser.add_argument('--continuation', default=False, type=bool, help='continue training')
    parser.add_argument('--plotsize', default=500, type=int, help='plotting size')
    parser.add_argument('--seed', default=233, type=int, help='random select samples for plotting')

    opt = parser.parse_args()

    if opt.layer != "":
        layers = list(map(int, opt.layer.split(',')))
    else:
        layers = []

    opt_lr = list(map(float, opt.lr.split(',')))
    opt_epochs = list(map(int, opt.epoch.split(',')))

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    flog = logging.FileHandler(opt.output + '/log.txt', mode='w')
    flog.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='[%(asctime)s] (%(levelname)s) %(message)s', datefmt='%Y-%d-%m %H:%M:%S')
    flog.setFormatter(formatter)
    clog = logging.StreamHandler()
    clog.setFormatter(formatter)
    logger.addHandler(flog)
    logger.addHandler(clog)

    if opt.featrm == 'auto':
        logger.info('Automatically remove features')
        featrm = [14, 15, 17, 18, 19, 20, 21, 22]
    elif opt.featrm == '':
        featrm = []
    else:
        featrm = list(map(int, opt.featrm.split(',')))
    logger.info('Remove Feature: %s' % featrm)

    logger.info('Reading data...')
    datax, datay, data_names = dataloader.load(filename=opt.input, target=opt.target, fps=opt.fp.split(','), featrm=featrm)

    logger.info('Selecting data...')
    selector = preprocessing.Selector(datax, datay, data_names)
    if opt.part:
        logger.info('Loading partition file %s' % opt.part)
        selector.load(opt.part)
    else:
        logger.warning("Partition file not found. Using auto-partition instead.")
        selector.partition(0.8, 0.1)
        selector.save(opt.output + '/part.txt')
    trainx, trainy, trainname = selector.training_set()
    validx, validy, validname = selector.validation_set()
    scaler = preprocessing.Scaler()
    scaler.fit(trainx)
    # scaler.save(opt.output + '/scale.txt')
    normed_trainx = scaler.transform(trainx)
    normed_validx = scaler.transform(validx)

    model = fitting.TorchMLPRegressor(len(trainx[0]), len(trainy[0]), layers, batch_size=opt.batch, batch_step=opt.step,
                                        is_gpu= False,
                                        args_opt={'optimizer'   : torch.optim.Adam,
                                                    'lr'          : opt.lr,
                                                    'weight_decay': opt.l2
                                                    }
                                        )
    model.load(opt.output + '/model.pt')

    size = 500

    data_smiles = [i.split('\t')[0] for i in validname]

    data_T = [i.split('\t')[1] for i in validname]

    data_P = [i.split('\t')[2] for i in validname]

    predy = model.predict_batch(torch.Tensor(normed_validx))

    error_y = abs(predy - validy).reshape(-1)

    draw(validy[:size], predy[:size], data_smiles[:size], data_T[:size], data_P[:size])

def DrawMol(mol,  kekulize=True):
    pattern = re.compile("<\?xml.*\?>")
    mc = Chem.MolFromSmiles(mol)
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        mc.Compute2DCoords()
    drawer = rdMolDraw2D.MolDraw2DSVG(100, 100)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    svg = re.sub(pattern, '', svg)
    return svg

    # data_svg = [ DrawMol(i) for i in data_smiles_plot[:size]]
    # SVG(data_svg[0])


def mapcolor(error, threshold=5): # map error to color
    if error > threshold:
        return  "#%02x%02x%02x" % (222,156, 83)
    else:
        return "#%02x%02x%02x" % (137,190,178)

def draw(validy, predy, smiles, T, P):
    error_y = abs(predy - validy)
    RE = error_y / validy * 100
    data_svg = [DrawMol(i) for i in smiles]
    colors = [ mapcolor(c) for c in RE ]
    TOOLS = "pan,wheel_zoom,reset, save, lasso_select"
    # TOOLTIPS = [ ("Smiles:", "@smiles"), ("T:", "@T"), ("P:", "@P"), ("log MSE:", "@error"), ("pic", "@svgs{safe}") ]
    TOOLTIPS = """
        <div>
            <div>@svgs{safe}</div>
            <div>Smiles: @smiles</div>
            <div>T: @T</div>
            <div>P: @P</div>
            <div>predy(g/cm^3): @y</div>
            <div>calc(g/cm^3): @x</div>
            <div>RelativeErr: @error %</div>
        </div>
        """
    data = ColumnDataSource({'x':validy, 'y':predy , 'smiles':smiles, 'T':T, 'P':P, 'error':RE, 'fill_color':colors, 'svgs':data_svg} )
    miny = float(min(validy))
    maxy = float(max(validy))
    p = figure( 
        title="result", tools=TOOLS,
        x_range=Range1d(0.9*miny,1.1*maxy), y_range=Range1d(0.9*miny,1.1*maxy)
    )
    p.select(LassoSelectTool).select_every_mousemove = False
    hover = HoverTool()
    hover.tooltips = TOOLTIPS
    p.tools.append(hover)
    stats = PreText(text='a', width=500)

    def selection_change(attrname, old, new):
        selected_idx = data.selected.indices
        selected_smiles=list(data.data['smiles'][idx])
        stats.text = smiles_str
    data.selected.on_change('indices', selection_change)

    p.line([miny, maxy], [miny, maxy],line_width=1)
    p.scatter('x', 'y', radius=0.002, source=data, fill_color='fill_color', fill_alpha=1, line_color=None)
    layout = row(p , stats)
    curdoc().add_root(layout)

    show(layout)

if __name__=="__main__":
    main()