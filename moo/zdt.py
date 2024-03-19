'''
Adding the _calc_pareto_set to the original pymoo implementation.
'''


from pymoo.problems.multi.zdt import ZDT1 as ZDT1_pymoo
from pymoo.problems.multi.zdt import ZDT2 as ZDT2_pymoo
from pymoo.problems.multi.zdt import ZDT3 as ZDT3_pymoo
from pymoo.problems.multi.zdt import ZDT4 as ZDT4_pymoo
from pymoo.problems.multi.zdt import ZDT6 as ZDT6_pymoo

import numpy as np

def calc_zdt_paret_set(n_var, n_pareto_points=100):
    '''
    Calculates the pareto set for all six ZDT problems.

    Parameters:
    ----------
    n_var (int): Number of decision variables.
    n_pareto_points (int): Number of points in the pareto set.

    Returns:
    --------
    np.ndarray: Pareto set.
    '''
    x_1 = np.linspace(0.0, 1.0, n_pareto_points)
    pareto_set = np.zeros((n_pareto_points, n_var))
    pareto_set[:, 0] = x_1
    return pareto_set

class ZDT1(ZDT1_pymoo):
    def _calc_pareto_set(self, n_pareto_points=100):
        return calc_zdt_paret_set(self.n_var, n_pareto_points)

class ZDT2(ZDT2_pymoo):
    def _calc_pareto_set(self, n_pareto_points=100):
        return calc_zdt_paret_set(self.n_var, n_pareto_points)

class ZDT3(ZDT3_pymoo):
    def _calc_pareto_set(self, n_pareto_points=100):
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]

        x_1 = []

        for r in regions:
            sample_current_region = np.linspace(r[0], r[1], int(n_pareto_points / len(regions)))
            x_1 = x_1 + sample_current_region.tolist()
        
        pareto_set = np.zeros((n_pareto_points, self.n_var))
        pareto_set[:, 0] = np.array(x_1)
        return pareto_set
        

class ZDT4(ZDT4_pymoo):
    def _calc_pareto_set(self, n_pareto_points=100):
        return calc_zdt_paret_set(self.n_var, n_pareto_points)

class ZDT6(ZDT6_pymoo):
    def _calc_pareto_set(self, n_pareto_points=1000):
        return calc_zdt_paret_set(self.n_var, n_pareto_points)

