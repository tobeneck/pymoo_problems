'''
Adding the _calc_pareto_set to the original pymoo implementation.
'''

from pymoo.problems.many.dtlz import DTLZ1 as DTLZ1_pymoo
from pymoo.problems.many.dtlz import DTLZ2 as DTLZ2_pymoo
from pymoo.problems.many.dtlz import DTLZ3 as DTLZ3_pymoo
from pymoo.problems.many.dtlz import DTLZ4 as DTLZ4_pymoo
from pymoo.problems.many.dtlz import DTLZ5 as DTLZ5_pymoo
from pymoo.problems.many.dtlz import DTLZ6 as DTLZ6_pymoo
from pymoo.problems.many.dtlz import DTLZ7 as DTLZ7_pymoo

import numpy as np

from pymoo.util.ref_dirs import get_reference_directions

def get_equidistant_2d_points(n_points):
    '''
    Returns equidistant points in 2D. The number of points is determined by the square root of n_points.

    Parameters:
    ----------
    n_points (int): Number of points.

    Returns:
    --------
    np.ndarray: Equidistant points.
    '''
    n_equidistant = np.floor(n_points ** (1/2)).astype(int)
    return np.array([[x, y] for x in np.linspace(0.0, 1.0, n_equidistant) for y in np.linspace(0.0, 1.0, n_equidistant)])

def get_equidistant_3d_points(n_points):
    '''
    Returns equidistant points in 3D. The number of points is determined by the cube root of n_points.

    Parameters:
    ----------
    n_points (int): Number of points.

    Returns:
    --------
    np.ndarray: Equidistant points.
    '''
    n_equidistant = np.floor(n_points ** (1/3)).astype(int)
    return np.array([[x, y, z] for x in np.linspace(0.0, 1.0, n_equidistant) for y in np.linspace(0.0, 1.0, n_equidistant) for z in np.linspace(0.0, 1.0, n_equidistant)])



def calc_dtlz_paret_set(n_var, n_obj, n_pareto_points=500, optimal_distance=0.5):
    '''
    Calculates the pareto set for all seven DTLZ problems.

    Parameters:
    ----------
    n_var (int): Number of decision variables.
    n_pareto_points (int): Number of points in the pareto set.

    Returns:
    --------
    np.ndarray: Pareto set.
    '''

    #first, compute the positional points
    n_positional_points = n_obj - 1
    #try to compute equidistant points for up to 4 objectives
    equidistant_points = []
    if n_positional_points == 1:
        equidistant_points = np.linspace(0.0, 1.0, n_pareto_points)
    elif n_positional_points == 2:
        equidistant_points = get_equidistant_2d_points(n_pareto_points)
    elif n_positional_points == 3:
        equidistant_points = get_equidistant_3d_points(n_pareto_points)
    
    #sample the rest randomly
    n_random_points = n_pareto_points - len(equidistant_points)
    random_points = np.random.rand(n_random_points, n_positional_points)

    #combine the points
    positional_x = np.concatenate([equidistant_points, random_points])


    #set x_1 and x_2
    pareto_set = np.ones((n_pareto_points, n_var)) * optimal_distance
    pareto_set[:, :n_obj-1] = positional_x

    return pareto_set

class DTLZ1(DTLZ1_pymoo):
    def _calc_pareto_set(self, n_pareto_points=500):
        return calc_dtlz_paret_set(self.n_var, self.n_obj, n_pareto_points)

class DTLZ2(DTLZ2_pymoo):
    def _calc_pareto_set(self, n_pareto_points=500):
        return calc_dtlz_paret_set(self.n_var, self.n_obj, n_pareto_points)
    
class DTLZ3(DTLZ3_pymoo):
    def _calc_pareto_set(self, n_pareto_points=500):
        return calc_dtlz_paret_set(self.n_var, self.n_obj, n_pareto_points)

class DTLZ5(DTLZ5_pymoo):
    def _calc_pareto_set(self, n_pareto_points=500):
        return calc_dtlz_paret_set(self.n_var, self.n_obj, n_pareto_points)

class DTLZ6(DTLZ6_pymoo):
    def _calc_pareto_set(self, n_pareto_points=500):
        return calc_dtlz_paret_set(self.n_var, self.n_obj, n_pareto_points, optimal_distance=0.0)