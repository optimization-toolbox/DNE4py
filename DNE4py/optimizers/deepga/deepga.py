
from .base import BaseGA
from .mutation import RealMutator
from .selection import TruncatedSelection


class TruncatedRealMutatorGA(TruncatedSelection, RealMutator, BaseGA):
    pass
