
""" Hand coded structral key indexer for alkanes
"""

from pybel import Molecule, Smarts
import numpy as np 


class CHSKIndexer:
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

    RB1_Matcher = Smarts('[R;CH1]([!H3])')
    RB2_Matcher = Smarts('[R;CH0]([!H3])([!H3])')
    RM1_Matcher = Smarts('[R;CH]([CH3])')
    RM2_Matcher = Smarts('[R;C]([CH3])([CH3])')

    # small rings

    R5_Matcher = Smarts('C1CCCC1')

    # special rings

    Fuse_Matcher = Smarts('[C;R2][C;R2]')
    Bridge_Matcher = Smarts('[C;R1][C;R1]([C;R1])[C;R1]([C;R1])[C;R1]')

    #
    Alkene1_Matcher = Smarts('[CH2]=C')
    Alkene2_Matcher = Smarts('C[CH1]=C')
    Alkene3_Matcher = Smarts('CC(C)=C')

    Alkene4_Matcher = Smarts('c[CH1]=C')
    Alkene5_Matcher = Smarts('cC(C)=C')
    Alkene6_Matcher = Smarts('cC(c)=C')

    rAlkene1_Matcher = Smarts('[CH]1=CCCC1')
    rAlkene2_Matcher = Smarts('[CH0]1=CCCC1')

    cAlkene1_Matcher = Smarts('C=[CH][CH]=C')
    cAlkene2_Matcher = Smarts('C=[CH][CH0]=C')
    cAlkene3_Matcher = Smarts('C=[CH0][CH0]=C')

    Alkyne1_Matcher = Smarts('[CH]#C')
    Alkyne2_Matcher = Smarts('CC#C')
    Alkyne3_Matcher = Smarts('cC#C')
    Alkyne4_Matcher = Smarts('C=CC#C')

    Benzene1_Matcher = Smarts('c1ccccc1')
    Benzene2_Matcher = Smarts('[cH0]1[cH1][cH1][cH1][cH1][cH1]1')
    Benzene3_Matcher = Smarts('[cH0]1[cH0][cH1][cH1][cH1][cH1]1')
    Benzene4_Matcher = Smarts('[cH0]1[cH1][cH0][cH1][cH1][cH1]1')
    Benzene5_Matcher = Smarts('[cH0]1[cH1][cH1][cH0][cH1][cH1]1')
    Benzene6_Matcher = Smarts('[cH0]1[cH0][cH0][cH1][cH1][cH1]1')
    Benzene7_Matcher = Smarts('[cH0]1[cH0][cH1][cH0][cH1][cH1]1')
    Benzene8_Matcher = Smarts('[cH0]1[cH1][cH0][cH1][cH0][cH1]1')
    Benzene09_Matcher = Smarts('[cH0]1[cH0][cH0][cH0][cH1][cH1]1')
    Benzene10_Matcher = Smarts('[cH0]1[cH0][cH0][cH1][cH0][cH1]1')
    Benzene11_Matcher = Smarts('[cH0]1[cH0][cH1][cH0][cH0][cH1]1')
    Benzene12_Matcher = Smarts('[cH0]1[cH0][cH0][cH0][cH0][cH1]1')
    Benzene13_Matcher = Smarts('[cH0]1[cH0][cH0][cH0][cH0][cH0]1')

    BenzeneC5_Matcher = Smarts('c1c(CCC2)c2ccc1')
    BenzeneC6_Matcher = Smarts('c1c(CCCC2)c2ccc1')

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

        chain4match = remove_repeat(CHSKIndexer.Chain4_Matcher.findall(molecule))
        chain6match = remove_repeat(CHSKIndexer.Chain6_Matcher.findall(molecule))

        myindex = [
            molecule.OBMol.NumHvyAtoms(),    # 0 NC
            len(hv[hv==4]),         # 1 C4
            len(hv[hv==3]),         # 2 C3
            len(hv[hv==2]),         # 3 C2
            len(hv[hv==1]),         # 4 C1
            len(chain4match),       # 5 chain4
            len(chain6match),       # 6 chain6
            len(CHSKIndexer.C3M1_Matcher.findall(molecule)),    # 7
            len(CHSKIndexer.C3M2_Matcher.findall(molecule)),    # 8
            len(CHSKIndexer.C4M1_Matcher.findall(molecule)),    # 9
            len(CHSKIndexer.C4M2_Matcher.findall(molecule)),    # 10
            len(CHSKIndexer.C4M3_Matcher.findall(molecule)),    # 11
            len(CHSKIndexer.RB1_Matcher.findall(molecule)),     # 12
            len(CHSKIndexer.RB2_Matcher.findall(molecule)),     # 13
            len(CHSKIndexer.RM1_Matcher.findall(molecule)),     # 14
            len(CHSKIndexer.RM2_Matcher.findall(molecule)),     # 15
            len(molecule.sssr),                                 # 16 Ring
            len(CHSKIndexer.Fuse_Matcher.findall(molecule)),    # 17 Fuse
            len(CHSKIndexer.Bridge_Matcher.findall(molecule)),  # 18 Bridge
            len(CHSKIndexer.Alkene1_Matcher.findall(molecule)),  # 19 Alkene
            len(CHSKIndexer.Alkene2_Matcher.findall(molecule)),  # 20 Alkene
            len(CHSKIndexer.Alkene3_Matcher.findall(molecule)),  # 21 Alkene
            len(CHSKIndexer.Alkene4_Matcher.findall(molecule)),  # 22 Alkene
            len(CHSKIndexer.Alkene5_Matcher.findall(molecule)),  # 23 Alkene
            len(CHSKIndexer.Alkene6_Matcher.findall(molecule)),  # 24 Alkene
            len(CHSKIndexer.Alkyne1_Matcher.findall(molecule)),  # 25 Alkyne
            len(CHSKIndexer.Alkyne2_Matcher.findall(molecule)),  # 26 Alkyne
            len(CHSKIndexer.Alkyne3_Matcher.findall(molecule)),  # 27 Alkyne
            len(CHSKIndexer.Alkyne4_Matcher.findall(molecule)),  # 27 Alkyne
            len(CHSKIndexer.Benzene1_Matcher.findall(molecule)),  # 28 Benzene ring
            len(CHSKIndexer.Benzene2_Matcher.findall(molecule)),  # 29 Benzene ring
            len(CHSKIndexer.Benzene3_Matcher.findall(molecule)),  # 30 Benzene ring
            len(CHSKIndexer.Benzene4_Matcher.findall(molecule)),  # 31 Benzene ring
            len(CHSKIndexer.Benzene5_Matcher.findall(molecule)),  # 32 Benzene ring
            len(CHSKIndexer.Benzene6_Matcher.findall(molecule)),  # 33 Benzene ring
            len(CHSKIndexer.Benzene7_Matcher.findall(molecule)),  # 34 Benzene ring
            len(CHSKIndexer.Benzene8_Matcher.findall(molecule)),  # 35 Benzene ring
            len(CHSKIndexer.Benzene09_Matcher.findall(molecule)),  # 36 Benzene ring
            len(CHSKIndexer.Benzene10_Matcher.findall(molecule)),  # 37 Benzene ring
            len(CHSKIndexer.Benzene11_Matcher.findall(molecule)),  # 38 Benzene ring
            len(CHSKIndexer.Benzene12_Matcher.findall(molecule)),  # 39 Benzene ring
            len(CHSKIndexer.Benzene13_Matcher.findall(molecule)),  # 40 Benzene ring
            len(CHSKIndexer.BenzeneC5_Matcher.findall(molecule)),  # 41 Benzene ring
            len(CHSKIndexer.BenzeneC6_Matcher.findall(molecule)),  # 42 Benzene ring
            len(CHSKIndexer.rAlkene1_Matcher.findall(molecule)),  # 43 Alkene
            len(CHSKIndexer.rAlkene2_Matcher.findall(molecule)),  # 44 Alkene
            len(CHSKIndexer.cAlkene1_Matcher.findall(molecule)),  # 45 Alkene
            len(CHSKIndexer.cAlkene2_Matcher.findall(molecule)),  # 46 Alkene
            len(CHSKIndexer.cAlkene3_Matcher.findall(molecule)),  # 47 Alkene
            ]

        return myindex

    def index(self, molecule:Molecule):

        return np.array(self._index_molecule(molecule))



