import numpy as np

from .base import BaseGA
from DNE4py.utils import MPIData
# class Ranking(BaseGA):
#     def ranking_initialize(self):
#         self.n_tasks = len(self.objective_function(self.members[0].phenotype))

#     def apply_ranking(self):

#         # Evaluate member:
#         self.cost_list = np.zeros((self.workers_per_rank, self.n_tasks)) #  (W,T) vs (W)
#         for i in range(len(self.cost_list)):
#             self.cost_list[i] = self.objective_function(self.members[i].phenotype)


#         # ======================= LOGGING =====================================
#         if self.verbose == 2:
#             self.logger.debug(f"\nPopulation Members (Initial):")
#             self.logger.debug(f"| index | seed | x0 | x1 | y |")
#             for index in range(len(self.members)):
#                 seed = self.members[index].genotype
#                 x0 = self.members[index].phenotype[0].round(10)
#                 x1 = self.members[index].phenotype[1].round(10)
#                 y = self.cost_list[index].round(10)
#                 self.logger.debug(f"| {index} | {seed} | {x0} | {x1} | {y} |")
#         # ===================== END LOGGING ===================================
#         # Save:
#         if (self.save > 0) and (self.generation % self.save == 0):
#             self.mpi_save(self.generation)

#         # Broadcast fitness:
#         cost_matrix = np.empty((self._size, self.workers_per_rank)) # (S,W,T) vs (S,W)
#         self._comm.Allgather([self.cost_list, self._MPI.FLOAT],
#                             [cost_matrix, self._MPI.FLOAT])

#         order = self.ordering(cost_matrix) # (S*W,T) vs (S*W)
#         rank = np.argsort(order)
#         ranks_and_members_by_performance = rank.reshape(self._size, self.workers_per_rank) #--#
        
#         return ranks_and_members_by_performance


#     def ordering(cost_matrix):
#         pass



class CompactRanking(BaseGA):
    def apply_ranking(self):

        # Evaluate member:
        self.cost_list = np.zeros(self.workers_per_rank)
        for i in range(len(self.cost_list)):
            self.cost_list[i] = self.objective_function(self.members[i].phenotype)


        # ======================= LOGGING =====================================
        if self.verbose == 2:
            self.logger.debug(f"\nPopulation Members (Initial):")
            self.logger.debug(f"| index | seed | x0 | x1 | y |")
            for index in range(len(self.members)):
                seed = self.members[index].genotype
                x0 = self.members[index].phenotype[0].round(10)
                x1 = self.members[index].phenotype[1].round(10)
                y = self.cost_list[index].round(10)
                self.logger.debug(f"| {index} | {seed} | {x0} | {x1} | {y} |")
        # ===================== END LOGGING ===================================
        # Save:
        if (self.save > 0) and (self.generation % self.save == 0):
            self.mpi_save(self.generation)

        # Broadcast fitness:
        cost_matrix = np.empty((self._size, self.workers_per_rank))
        self._comm.Allgather([self.cost_list, self._MPI.FLOAT],
                             [cost_matrix, self._MPI.FLOAT])

        # Truncate Selection (broadcast genotypes and update members):
        order = np.argsort(cost_matrix.flatten())
        rank = np.argsort(order)
        ranks_and_members_by_performance = rank.reshape(cost_matrix.shape)
        return ranks_and_members_by_performance


class CompositeRanking(BaseGA):

    def apply_ranking(self):
        
        # Evaluate member:
        self.cost_list = [] #--#
        for i in range(self.workers_per_rank):
            self.cost_list.append(self.objective_function(self.members[i].phenotype))
        self.cost_list = np.array(self.cost_list)
        self.n_tasks = self.cost_list.shape[1] 

        # ======================= LOGGING =====================================
        if self.verbose == 2:
            self.logger.debug(f"\nPopulation Members (Initial):")
            self.logger.debug(f"| index | seed | x0 | x1 | y |")
            for index in range(len(self.members)):
                seed = self.members[index].genotype
                x0 = self.members[index].phenotype[0].round(10)
                x1 = self.members[index].phenotype[1].round(10)
                y = self.cost_list[index].round(10)
                self.logger.debug(f"| {index} | {seed} | {x0} | {x1} | {y} |")
        # ===================== END LOGGING ===================================



        # Broadcast fitness:
        cost_matrix = np.empty((self._size, self.workers_per_rank, self.n_tasks)) #--#
        self._comm.Allgather([self.cost_list, self._MPI.FLOAT],
                             [cost_matrix, self._MPI.FLOAT])

        # Compute "on instances" average rankings   
        average_ranking = self.ranking(cost_matrix)

        # Save:
        if (self.save > 0) and (self.generation % self.save == 0):
            self.mpi_save(self.generation)

            # add the specific ranking data
            self.mpidata_rankings.write(average_ranking)

        # Truncate Selection (broadcast genotypes and update members):
        order = np.argsort(average_ranking) #--#
        rank = np.argsort(order)
        ranks_and_members_by_performance = rank.reshape(self._size, self.workers_per_rank) #--#
        return ranks_and_members_by_performance



    def ranking(self, cost_matrix):
        """
        @TODO : gèrer les ex-aequo
        """
        data = cost_matrix.reshape(self._size * self.workers_per_rank, self.n_tasks)
        
        for inst in range(data.shape[1]):
            data[:,inst] = rankdata(data[:,inst]) 
        return np.mean(data, axis=1)


# Move somewhere else
# https://github.com/numbbo/coco/blob/master/code-postprocessing/cocopp/toolsstats.py
def rankdata(a):
    """Ranks the data in a, dealing with ties appropriately.
    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.
    Example:
      In [15]: stats.rankdata([0, 2, 2, 3])
      Out[15]: array([ 1. ,  2.5,  2.5,  4. ])
    Parameters:
      - *a* : array
        This array is first flattened.
    Returns:
      An array of length equal to the size of a, containing rank scores.
    """
    a = np.ravel(a)
    n = len(a)
    svec, ivec = fastsort(a)
    sumranks = 0
    dupcount = 0
    newarray = np.zeros(n, float)
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i == n - 1 or svec[i] != svec[i + 1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i - dupcount + 1, i + 1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray
# Dimensions
# Calcul des ranks (non trivial dans le cas composite)

def fastsort(a):
     it = np.argsort(a)
     as_ = a[it]
     return as_, it