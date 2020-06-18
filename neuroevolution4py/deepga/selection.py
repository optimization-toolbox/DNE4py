import numpy as np

from .base import BaseGA

class TruncatedSelection(BaseGA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Input:
        self._num_parents = kwargs.get('num_parents')

        # Init:
        self.rgn_selection = np.random.RandomState(self.global_seed)

        if self.num_elite > self._num_parents:
            raise AssertionError("Number of elite has to be less than the"
                                 " number of parents")

    def apply_selection(self, ranks_and_members_by_performance):

        # 1 - Define parents and no parents:
        parents_mask = ranks_and_members_by_performance < self._num_parents
        no_parents_mask = np.invert(parents_mask)

        row, column = np.where(parents_mask)
        parents_indexes = tuple(zip(row, column))
        parent_options = range(len(parents_indexes))

        row, column = np.where(no_parents_mask)
        no_parents_indexes = tuple(zip(row, column))

        # 2 - Update member for each rank:
        # Build message_list:
        messenger_list = []
        for dest_rank, dest_member_id in no_parents_indexes:
            choice = self.rgn_selection.choice(parent_options)
            src_rank, src_member_id = parents_indexes[choice]
            messenger_list.append((src_rank,
                                   src_member_id,
                                   dest_rank,
                                   dest_member_id))

        # Dispatch message_list:
        for (src_rank,
             src_member_id,
             dest_rank,
             dest_member_id) in messenger_list:
            if src_rank != dest_rank:  # Need MPI send
                if self._rank == src_rank:
                    message = (self.members[src_member_id].genotype,
                               dest_member_id)
                    self._comm.send(message, dest=dest_rank)
                elif self._rank == dest_rank:
                    new_genotype, member_id = self._comm.recv(source=src_rank)
                    self.members[member_id].recreate(new_genotype)
            else:
                if (self._rank == src_rank):
                    new_genotype = self.members[src_member_id].genotype
                    member_id = dest_member_id
                    self.members[member_id].recreate(new_genotype)
