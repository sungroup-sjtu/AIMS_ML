from mdlearn.fp import Fingerprint
from .mol_io.molecule import Atom, Molecule


class Team(Fingerprint):
    def __init__(self):
        super().__init__()

    def replace_type_neigh(self, type):
        d = {
            # 'c_4'   : 'c_4',
            # 'c_4h'  : 'c_4h',
            # 'c_4h2' : 'c_4h2',
            # 'c_4h3' : 'c_4h3',
            # 'c_43'  : 'c_4r',
            # 'c_43h' : 'c_4rh',
            # 'c_43h2': 'c_4rh2',
            # 'c_44'  : 'c_4r',
            # 'c_44h' : 'c_4rh',
            # 'c_44h2': 'c_4rh2',
            # 'c_45'  : 'c_4r',
            # 'c_45h' : 'c_4rh',
            # 'c_45h2': 'c_4rh2',
        }
        return d.get(type, type)

    def replace_type_center(self, type):
        d = {
        }
        return d.get(type, type)

    def calc_atom(self, atom: Atom, radius: int) -> str:
        if radius == 0:
            return '0.' + atom.type

        if radius == 1:
            types_neigh = [self.replace_type_neigh(a.type) for a in atom.neighbours]
            types_neigh.sort()
            return '1.' + ','.join([self.replace_type_center(atom.type)] + types_neigh)

        if radius == 2:
            raise Exception('radius > 1 not supported')

    def calc_molecule(self, molecule: Molecule, radius_list: [int]):
        fps = {}
        for r in radius_list:
            for atom in molecule.atoms:
                fp = self.calc_atom(atom, r)
                if fp not in fps:
                    fps[fp] = 0
                fps[fp] += 1

        self.bit_count = fps

    @staticmethod
    def merge_team_fps(fps: [Fingerprint], limit=1) -> [Fingerprint]:
        idx_set = set()
        fps_new = []
        for fp in fps:
            idx_set.update(fp.bit_count.keys())

        idx_list = list(sorted(idx_set))

        ### Filter idx based on times
        print('%i identifiers total' % (len(idx_list)))
        idx_times = dict([(k, 0) for k in idx_list])
        for fp in fps:
            for k, v in fp.bit_count.items():
                idx_times[k] += 1
        idx_list = [k for (k, v) in idx_times.items() if k.startswith('0') or v >= limit]
        print('%i identifiers saved' % (len(idx_list)))

        for fp in fps:
            bit_count = dict([(k, 0) for k in idx_list])
            for k, v in fp.bit_count.items():
                if k in idx_list:
                    bit_count[k] = v

            fp_new = Fingerprint()
            fp_new.bit_count = bit_count
            fps_new.append(fp_new)

        return fps_new
