
""" Hand coded structral key indexer for alkanes
"""

import pybel
from pybel import Smarts
import numpy as np


class SimpleIndexer:
    """ Structure key indexer
    """
    name = 'simple'

    # chain

    Chain4_Matcher = Smarts('CCCC')
    Chain6_Matcher = Smarts('CCCCCC')

    # methyl

    C3M1_Matcher = Smarts('[CH3][CH]([!H3])[!H3]')
    C3M2_Matcher = Smarts('[CH3][CH]([CH3])[!H3]')
    C4M1_Matcher = Smarts('[CH3]C([!H3])([!H3])[!H3]')
    C4M2_Matcher = Smarts('[!H3]C([CH3])([CH3])[!H3]')
    C4M3_Matcher = Smarts('[CH3]C([CH3])([CH3])[!H3]')

    # ring branches

    RB1_Matcher = Smarts('[RH1]([!H3])')
    RB2_Matcher = Smarts('[RH0]([!H3])([!H3])')
    RM1_Matcher = Smarts('[RH]([CH3])')
    RM2_Matcher = Smarts('[R]([CH3])([CH3])')

    # small rings

    R3_Matcher = Smarts('C1CC1')
    R4_Matcher = Smarts('C1CCC1')
    R5_Matcher = Smarts('C1CCCC1')
    R6_Matcher = Smarts('C1CCCCC1')
    R7_Matcher = Smarts('C1CCCCCC1')
    R8_Matcher = Smarts('C1CCCCCCC1')

    # special rings

    Spiro_Matcher = Smarts('[R1][R2]([R1])([R1])[R1]')
    Fuse_Matcher = Smarts('[R2][R2]')
    Bridge_Matcher = Smarts('[R1][R1]([R1])[R1]([R1])[R1]')
    Bridge3_Matcher = Smarts('[R3]')

    def __init__(self, *args):
        pass

    def _index_smiles(self, smiles):
        molecule = pybel.readstring('smi', smiles)

        hv = np.array([a.heavyvalence for a in molecule.atoms])

        def remove_repeat(l):
            ret = []
            has_set = set()
            for l_ in l:
                if not has_set.intersection(l_):
                    ret.append(l_)
                    has_set.update(l_)

            return ret

        chain4match = remove_repeat(SimpleIndexer.Chain4_Matcher.findall(molecule))
        chain6match = remove_repeat(SimpleIndexer.Chain6_Matcher.findall(molecule))

        myindex = [
            molecule.OBMol.NumHvyAtoms(),    # 0 NC
            len(chain4match),       # 5 chain4
            len(chain6match),       # 6 chain6
            len(molecule.sssr),                                # 16 Ring
            len(SimpleIndexer.R3_Matcher.findall(molecule)),      # 17 R3
            len(SimpleIndexer.R4_Matcher.findall(molecule)),      # 18 R4
            len(SimpleIndexer.R5_Matcher.findall(molecule)),      # 18 R4
            len(SimpleIndexer.R6_Matcher.findall(molecule)),      # 18 R4
            len(SimpleIndexer.R7_Matcher.findall(molecule)),      # 18 R4
            len(SimpleIndexer.R8_Matcher.findall(molecule)),      # 18 R4
            len(SimpleIndexer.Spiro_Matcher.findall(molecule)),    # 20 Fuse
            len(SimpleIndexer.Fuse_Matcher.findall(molecule)),    # 20 Fuse
            len(SimpleIndexer.Bridge_Matcher.findall(molecule)),  # 21 Bridge
            ]

        return myindex

    def index(self, smiles):
        return np.array(self._index_smiles(smiles))

    def index_list(self, smiles_list):
        return [self.index(s) for s in smiles_list]
