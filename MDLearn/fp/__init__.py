from .wyz import WyzIndexer
from .simple import SimpleIndexer
from .rdk import ECFP4Indexer, MorganCountIndexer, Morgan1CountIndexer

encoders_dict = dict([(e.name, e) for e in (WyzIndexer, SimpleIndexer,
                                            ECFP4Indexer, MorganCountIndexer, Morgan1CountIndexer,
                                            )])


class Fingerprint():
    def __init__(self):
        self.bit_count = {}

    @property
    def idx(self):
        return list(self.bit_count.keys())

    @property
    def bits(self):
        return list(self.bit_count.values())
