class Atom():
    def __init__(self):
        self.name = None
        self.type = None
        self.element = None
        self.charge = None

        self.bonds: [Bond] = []
        self.neighbours: [Atom] = []

    @property
    def is_H(self):
        return self.element == 'H'

    @property
    def n_bond(self):
        return len(self.bonds)

    @property
    def n_neigh_H(self):
        return len([a for a in self.neighbours if a.is_H])

    @property
    def n_neigh_heavy(self):
        return len([a for a in self.neighbours if not a.is_H])

    def __repr__(self):
        return '<Atom: %s %s>' % (self.name, self.type)


class Bond():
    def __init__(self, atom1: Atom, atom2: Atom):
        self.atom1 = atom1
        self.atom2 = atom2
        atom1.bonds.append(self)
        atom2.bonds.append(self)
        atom1.neighbours.append(atom2)
        atom2.neighbours.append(atom1)

    def __repr__(self):
        return '<Bond: %s %s>' % (self.atom1.name, self.atom2.name)


class Molecule():
    def __init__(self, atoms: [Atom]):
        self.atoms: [Atom] = atoms

    def canonicalize(self):
        pass
