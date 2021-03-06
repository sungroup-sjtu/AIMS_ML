from .fp import Fingerprint
from .wyz import WyzIndexer
from .simple import SimpleIndexer, ExtraIndexer
from .rdk import *

encoders_dict = dict([(e.name, e) for e in (WyzIndexer, SimpleIndexer, ExtraIndexer,
                                            ECFP4Indexer, MorganCountIndexer, Morgan1CountIndexer,
                                            PredefinedMorganCountIndexer, PredefinedMorgan1CountIndexer,
                                            TopologicalCountIndexer, PredefinedTopologicalCountIndexer
                                            )])
