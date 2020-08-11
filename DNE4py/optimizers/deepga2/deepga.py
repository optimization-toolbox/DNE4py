
from .base import BaseGA
from .mutation import RealMutator
from .selection import TruncatedSelection
from .ranking import CompactRanking, CompositeRanking


class TruncatedRealMutatorCompactGA(TruncatedSelection, RealMutator, CompactRanking, BaseGA):
    pass

class TruncatedRealMutatorCompositeGA(TruncatedSelection, RealMutator, CompositeRanking, BaseGA):
    pass
