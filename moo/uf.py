import numpy as np
from pymoo.core.problem import Problem

from .lz import LZ2, LZ5, LZ6, LZ8

"""
Implementation of the UF problem family as defined in:
Q. Zhang, A. Zhou, S. Zhao, P. Suganthan, W. Liu, und S. Tiwari, „Multiobjective optimization Test Instances for the CEC 2009 Special Session and Competition“, Mechanical Engineering, Jan. 2008.
"""

class UF(Problem):

    def __init__(
        self,
        xl,
        xu,
        n_var: int,
        n_obj: int,
    ):
        """UF benchmark family as defined in:
        Q. Zhang, A. Zhou, S. Zhao, P. Suganthan, W. Liu, und S. Tiwari,
        „Multiobjective optimization Test Instances for the CEC 2009 Special Session and Competition“,
        Mechanical Engineering, Jan. 2008.

        Some functions are doubled from the LZ benchmark family.
        """
        super().__init__(n_var=n_var,
                             n_obj=n_obj,
                             xl=xl,
                             xu=xu,
                             elementwise=True,
                             )
    
    def _calc_pareto_front(self, n_pareto_points=500):
        return self.evaluate(self._calc_pareto_set(n_pareto_points))

class UF1(LZ2): #exatly the same as LZ2
    pass

class UF2(LZ5): #exatly the same as LZ5
    pass

class UF3(LZ8): #exatly the same as LZ6
    pass

class UF8(LZ6): #exatly the same as LZ6
    pass

class UF9(UF):
    def __init__(self, n_var=30, epsilon=0.1):
        xl=np.ones(n_var) * -2
        xu=np.ones(n_var) * 2
        xl[0] = 0.0
        xl[1] = 0.0
        xu[0] = 1.0
        xu[1] = 1.0

        self.epsilon = epsilon
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=3,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        j_1 = np.arange(3, self.n_var, 3)
        j_2 = np.arange(4, self.n_var, 3)
        j_3 = np.arange(2, self.n_var, 3)

        f_1_sum = np.sum( [np.power( x[i] - 2*x[1]*np.sin(2*np.pi*x[0] + ( (i+1)*np.pi/self.n_var ) ), 2 ) for i in j_1] )
        f_2_sum = np.sum( [np.power( x[i] - 2*x[1]*np.sin(2*np.pi*x[0] + ( (i+1)*np.pi/self.n_var ) ), 2 ) for i in j_2] )
        f_3_sum = np.sum( [np.power( x[i] - 2*x[1]*np.sin(2*np.pi*x[0] + ( (i+1)*np.pi/self.n_var ) ), 2 ) for i in j_3] )

        f_1 = 0.5 * ( max([ 0, (1+self.epsilon)*(1-4*np.power(2*x[0]-1,2)) ]) +2*x[0] ) *  x[1] + (2/len(j_1)) * f_1_sum
        f_2 = 0.5 * ( max([ 0, (1+self.epsilon)*(1-4*np.power(2*x[0]-1,2)) ]) -2*x[0]+2 ) *  x[1] + (2/len(j_2)) * f_2_sum
        f_3 = 1 - x[1] + (2/len(j_3)) * f_3_sum

        out["F"] = [f_1, f_2, f_3]

    def _calc_pareto_set(self, n_pareto_points=625):
        #calculate the optimal values for x_1 and x_2. Try to get as many equidistant points as possible and calculate the rest randonly
        n_equidistant_samples = np.floor( np.sqrt(n_pareto_points) ).astype(int)
        half_n_equidistant_samples = int(n_equidistant_samples / 2)
        n_equidistant_samples = int(half_n_equidistant_samples * 2) #has to be dividable by two for the two intervals of x_1
        n_random_samples = n_pareto_points - (n_equidistant_samples * n_equidistant_samples)

        x_1_set = np.append( np.linspace(0.0, 0.25, half_n_equidistant_samples), np.linspace(0.75, 1.0, half_n_equidistant_samples) )
        equidistant_samples = np.array( [[x, y] for x in x_1_set for y in np.linspace(0.0, 1.0, n_equidistant_samples)] )
        

        #Generate random samples. Make shure to generate random numbers only in the intervals for x_1
        random_x_1 = np.random.random(n_random_samples) * 0.5
        random_x_1[random_x_1 > 0.25] += 0.75-0.25
        random_x2 = np.random.rand(n_random_samples)
        random_samples = np.zeros((n_random_samples, 2))
        random_samples[:,0] = random_x_1
        random_samples[:,1] = random_x2
        x_1_2 = np.vstack((equidistant_samples, random_samples)) #the optimal values for x[:,0] and x[:,1]

        #set x_1 and x_2
        pareto_set = np.zeros((n_pareto_points, self.n_var))
        pareto_set[:, 0] = x_1_2[:,0]
        pareto_set[:, 1] = x_1_2[:,1]

        #calculate the remaining values
        for j in range(2, self.n_var):
            pareto_set[:, j] = 2*pareto_set[:,1] * np.sin(2*np.pi*pareto_set[:,0] + ( (j+1)*np.pi/self.n_var ) )

        return pareto_set

class UF10(UF8):#PS and bounds are the same as UF8
    def _evaluate(self, x, out, *args, **kwargs):
        j_1 = np.arange(3, self.n_var, 3)
        j_2 = np.arange(4, self.n_var, 3)
        j_3 = np.arange(2, self.n_var, 3)

        def y_i(x, j):
            return x[j] - 2*x[1]*np.sin(2*np.pi*x[0] + ((j+1)*np.pi)/self.n_var)

        f_1_sum = np.sum( [4*np.power(y_i(x,i), 2) - np.cos(8*np.pi*y_i(x,i)) + 1 for i in j_1] )
        f_2_sum = np.sum( [4*np.power(y_i(x,i), 2) - np.cos(8*np.pi*y_i(x,i)) + 1 for i in j_2] )
        f_3_sum = np.sum( [4*np.power(y_i(x,i), 2) - np.cos(8*np.pi*y_i(x,i)) + 1 for i in j_3] )

        f_1 = np.cos(0.5*np.pi*x[0]) * np.cos(0.5*np.pi*x[1]) + (2/len(j_1)) * f_1_sum
        f_2 = np.cos(0.5*np.pi*x[0]) * np.sin(0.5*np.pi*x[1]) + (2/len(j_2)) * f_2_sum
        f_3 = np.sin(0.5*np.pi*x[0]) + (2/len(j_3)) * f_3_sum

        out["F"] = [f_1, f_2, f_3]