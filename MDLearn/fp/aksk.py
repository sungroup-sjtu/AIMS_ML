
""" Hand coded structral key indexer for alkanes
"""

from pybel import Molecule, Smarts
import numpy as np 


class AlkaneSKIndexer:
    """ Structure key indexer 
    """

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
    R8_Matcher = Smarts('C1CCCCCCC1')

    # special rings

    Fuse_Matcher = Smarts('[R2][R2]')
    Bridge_Matcher = Smarts('[R1][R1]([R1])[R1]([R1])[R1]')
    Bridge3_Matcher = Smarts('[R3]')

    def __init__(self, *args):
        super().__init__(*args)

    def _index_molecule(self, molecule:Molecule):
        hv = np.array([a.heavyvalence for a in molecule.atoms])

        def remove_repeat(l):
            ret = []
            has_set = set()
            for l_ in l:
                if not has_set.intersection(l_):
                    ret.append(l_)
                    has_set.update(l_)

            return ret 

        chain4match = remove_repeat(AlkaneSKIndexer.Chain4_Matcher.findall(molecule))
        chain6match = remove_repeat(AlkaneSKIndexer.Chain6_Matcher.findall(molecule))

        myindex = [
            len(molecule.atoms),    # 0 NC
            len(hv[hv==4]),         # 1 C4
            len(hv[hv==3]),         # 2 C3
            len(hv[hv==2]),         # 3 C2
            len(hv[hv==1]),         # 4 C1
            len(chain4match),       # 5 chain4
            len(chain6match),       # 6 chain6
            len(AlkaneSKIndexer.C3M1_Matcher.findall(molecule)),    # 7
            len(AlkaneSKIndexer.C3M2_Matcher.findall(molecule)),    # 8
            len(AlkaneSKIndexer.C4M1_Matcher.findall(molecule)),    # 9
            len(AlkaneSKIndexer.C4M2_Matcher.findall(molecule)),    # 10
            len(AlkaneSKIndexer.C4M3_Matcher.findall(molecule)),    # 11
            len(AlkaneSKIndexer.RB1_Matcher.findall(molecule)),     # 12
            len(AlkaneSKIndexer.RB2_Matcher.findall(molecule)),     # 13    
            len(AlkaneSKIndexer.RM1_Matcher.findall(molecule)),     # 14
            len(AlkaneSKIndexer.RM2_Matcher.findall(molecule)),     # 15
            len(molecule.sssr),                                     # 16 Ring
            len(AlkaneSKIndexer.R3_Matcher.findall(molecule)),      # 17 R3
            len(AlkaneSKIndexer.R4_Matcher.findall(molecule)),      # 18 R4
            len(AlkaneSKIndexer.R8_Matcher.findall(molecule)),      # 19 R8
            len(AlkaneSKIndexer.Fuse_Matcher.findall(molecule)),    # 20 Fuse
            len(AlkaneSKIndexer.Bridge_Matcher.findall(molecule)),  # 21 Bridge
            len(AlkaneSKIndexer.Bridge3_Matcher.findall(molecule))  # 22 Bridge3
            ]

        return myindex

    def index(self, molecule:Molecule):

        return np.array(self._index_molecule(molecule))
