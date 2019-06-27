import pybel
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol
import numpy as np

from . import Fingerprint


class SimpleIndexer(Fingerprint):
    name = 'simple'

    # small rings
    r3_Matcher = pybel.Smarts('*1**1')
    r4_Matcher = pybel.Smarts('*1***1')
    r5_Matcher = pybel.Smarts('*1****1')
    r6_Matcher = pybel.Smarts('*1*****1')
    r7_Matcher = pybel.Smarts('*1******1')
    r8_Matcher = pybel.Smarts('*1*******1')

    # special rings
    RR_Matcher = pybel.Smarts('[R]!@[R]')
    R_R_Matcher = pybel.Smarts('[R]!@*!@[R]')

    def __init__(self, *args):
        super().__init__()
        pass

    def get_chain_length(self, rdk_mol: Mol):
        wiener = 0
        max_shortest = 0
        mol = Chem.RemoveHs(rdk_mol)
        n_atoms = mol.GetNumAtoms()
        for i in range(0, n_atoms):
            for j in range(i + 1, n_atoms):
                shortest = len(Chem.GetShortestPath(mol, i, j)) - 1
                wiener += shortest
                max_shortest = max(max_shortest, shortest)
        return int(np.log(wiener) * 10), max_shortest

    def get_multiring_atoms_bonds(self, rdk_mol: Mol, smiles):
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
        index = [py_mol.OBMol.NumHvyAtoms(),
                 self.get_chain_length(rd_mol)[1]
                 ] + \
                [
                    Chem.CalcNumRotatableBonds(rd_mol),
                    len(py_mol.sssr),
                    len(SimpleIndexer.r3_Matcher.findall(py_mol)) +
                    len(SimpleIndexer.r4_Matcher.findall(py_mol)),
                    len(SimpleIndexer.r5_Matcher.findall(py_mol)),
                    len(SimpleIndexer.r6_Matcher.findall(py_mol)),
                    len(SimpleIndexer.r7_Matcher.findall(py_mol)) +
                    len(SimpleIndexer.r8_Matcher.findall(py_mol)),
                    Chem.CalcNumAromaticRings(rd_mol),
                    len(SimpleIndexer.RR_Matcher.findall(py_mol)),  # Linked rings
                    len(SimpleIndexer.R_R_Matcher.findall(py_mol)),  # Separated rings
                ] + \
                list(self.get_multiring_atoms_bonds(rd_mol, smiles))

        return np.array(index)

    def index_list(self, smiles_list):
        return [self.index(s) for s in smiles_list]
