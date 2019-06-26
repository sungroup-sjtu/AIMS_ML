import sys
import pybel
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol
import numpy as np

from . import Fingerprint


class SimpleIndexer(Fingerprint):
    name = 'simple'

    # bridged atoms
    bridg_Matcher = pybel.Smarts('[x3]')
    # spiro atoms
    spiro_Matcher = pybel.Smarts('[x4]')
    # linked rings
    RR_Matcher = pybel.Smarts('[R]!@[R]')
    # separated rings
    R_R_Matcher = pybel.Smarts('[R]!@*!@[R]')

    def __init__(self):
        super().__init__()
        pass

    def get_shortest_wiener(self, rdk_mol: Mol):
        wiener = 0
        max_shortest = 0
        mol = Chem.RemoveHs(rdk_mol)
        n_atoms = mol.GetNumAtoms()
        for i in range(0, n_atoms):
            for j in range(i + 1, n_atoms):
                shortest = len(Chem.GetShortestPath(mol, i, j)) - 1
                wiener += shortest
                max_shortest = max(max_shortest, shortest)
        return max_shortest, int(np.log(wiener) * 10)

    def get_ring_info(self, py_mol):
        r34 = 0
        r5 = 0
        r6 = 0
        r78 = 0
        rlt8 = 0
        aro = 0
        for r in py_mol.sssr:
            rsize = r.Size()
            if rsize == 3 or rsize == 4:
                r34 += 1
            elif r.IsAromatic():
                aro += 1
            elif rsize == 5:
                r5 += 1
            elif rsize == 6:
                r6 += 1
            elif rsize == 7 or rsize == 8:
                r78 += 1
            else:
                rlt8 += 1

        return r34, r5, r6, r78, rlt8, aro

    def get_multiring_atoms_bonds(self, rdk_mol: Mol, smiles):
        '''
        Not used
        '''
        atom_ring_times = [0] * rdk_mol.GetNumAtoms()
        bond_ring_times = [0] * rdk_mol.GetNumBonds()

        # TODO GetRingInfo gives SymmetricSSSR, not TRUE SSSR
        ri = rdk_mol.GetRingInfo()
        for id_atoms in ri.AtomRings():
            for ida in id_atoms:
                atom_ring_times[ida] += 1
        for id_bonds in ri.BondRings():
            for idb in id_bonds:
                bond_ring_times[idb] += 1

        n_atoms_multiring = len(list(filter(lambda x: x > 1, atom_ring_times)))
        n_bonds_multiring = len(list(filter(lambda x: x > 1, bond_ring_times)))

        py_mol = pybel.readstring('smi', smiles)
        if ri.NumRings() != len(py_mol.sssr):
            print('WARNING: SymmetricSSSR not equal to TRUE SSSR in rdkit. Use Openbabel instead:', smiles)
            n_atoms_multiring = pybel.Smarts('[R2]').findall(py_mol).__len__()
            n_bonds_multiring = n_atoms_multiring - 1

        return n_atoms_multiring, n_bonds_multiring

    def index(self, smiles):
        rd_mol: Mol = Chem.MolFromSmiles(smiles)
        py_mol = pybel.readstring('smi', smiles)
        index = [
                    py_mol.OBMol.NumHvyAtoms(),
                    int(round(py_mol.molwt, 1) * 10),
                    self.get_shortest_wiener(rd_mol)[0],
                    Chem.CalcNumRotatableBonds(Chem.AddHs(rd_mol)),
                    len(SimpleIndexer.bridg_Matcher.findall(py_mol)),
                    len(SimpleIndexer.spiro_Matcher.findall(py_mol)),
                    len(SimpleIndexer.RR_Matcher.findall(py_mol)),
                    len(SimpleIndexer.R_R_Matcher.findall(py_mol)),
                ] + \
                list(self.get_ring_info(py_mol))

        return np.array(index)

    def index_list(self, smiles_list):
        if self._silent:
            return [self.index(s) for s in smiles_list]

        l = []
        print('Calculate ...')
        for i, s in enumerate(smiles_list):
            if i % 100 == 0:
                sys.stdout.write('\r\t%i' % i)
            l.append(self.index(s))
        print('')

        return l
