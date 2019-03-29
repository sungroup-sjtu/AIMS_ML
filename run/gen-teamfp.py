#!/usr/bin/env python3

import os
import sys
import argparse
import base64
import pandas as pd

sys.path.append('..')
from mdlearn.teamfp.mol_io import Msd
from mdlearn.teamfp.mol_io import Molecule
from mdlearn.teamfp import Team

parser = argparse.ArgumentParser(description='Generate fingerprints')
parser.add_argument('-i', '--input', type=str, help='Data')
parser.add_argument('-o', '--output', default='fp', help='Output directory')

opt = parser.parse_args()

df = pd.read_csv(opt.input, sep='\s+', header=0)
smiles_list = df.SMILES.unique().tolist()

if not os.path.exists(opt.output):
    os.mkdir(opt.output)


def read_msd(msd_file) -> Molecule:
    msd = Msd()
    msd.read(msd_file, ignore_H=True)
    return msd.molecule


if __name__ == '__main__':
    fps = []
    for smiles in smiles_list:
        mol = read_msd('msdfiles/%s.msd' % base64.b64encode(smiles.encode()).decode())
        fp = Team()
        fp.calc_molecule(mol, radius_list=[0,1])
        fps.append(fp)

    fps_new = Team.merge_team_fps(fps, limit=40)

    print(fps_new[0].idx)

    with open(os.path.join(opt.output, 'fp_team'), 'w') as f:
        for i, fp in enumerate(fps_new):
            f.write('%s %s\n' % (smiles_list[i], ','.join(map(str, fp.bits))))
