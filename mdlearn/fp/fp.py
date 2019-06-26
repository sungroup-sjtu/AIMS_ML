class Fingerprint():
    def __init__(self):
        self.bit_count = {}
        self.use_pre_idx_list = False
        self._silent = False

    @property
    def idx_list(self):
        return list(self.bit_count.keys())

    @property
    def bit_list(self):
        return list(self.bit_count.values())
