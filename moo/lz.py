import numpy as np
from pymoo.core.problem import Problem

"""
Implementation of the LZ problem family as defined in:

H. Li and Q. Zhang. Multiobjective optimization problems with complicated pareto sets, MOEA/D and NSGA-II.
IEEE Transactions on Evolutionary Computation, 12(2):284-302, April 2009.
"""

class LZ(Problem):

    def __init__(
        self,
        xl,
        xu,
        n_var: int,
        n_obj: int,
    ):
        """LZ benchmark family as defined in:
        H. Li and Q. Zhang. Multiobjective optimization problems with complicated pareto sets, MOEA/D and NSGA-II.
        IEEE Transactions on Evolutionary Computation, 12(2):284-302, April 2009.
        """
        super().__init__(n_var=n_var,
                             n_obj=n_obj,
                             xl=xl,
                             xu=xu,
                             elementwise=True,
                             )
    
    def _calc_pareto_front(self, n_pareto_points=500):
        return self.evaluate(self._calc_pareto_set(n_pareto_points))


class LZ1(LZ):
    def __init__(self, n_var=10):
        xl=0.0
        xu=1.0
        super().__init__(
            n_var=n_var, 
            xl=xl,
            xu=xu,
            n_obj=2,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        odd = np.arange(1, self.n_var, 2)
        even = np.arange(2, self.n_var, 2) #zero is not included

        f_1_sum = np.sum( [np.power( x[i] - np.power( x[0], 0.5 * ( 1.0 + ( (3*(i-1) / (self.n_var - 2) ) ) ) ), 2 ) for i in odd] )
        f_2_sum = np.sum( [np.power( x[i] - np.power( x[0], 0.5 * ( 1.0 + ( (3*(i-1) / (self.n_var - 2) ) ) ) ), 2 ) for i in even] )

        f_1 = x[0] + (1 / len(odd)) * f_1_sum
        f_2 = 1 - np.sqrt(x[0]) + (1 / len(even)) * f_2_sum
        out["F"] = [f_1, f_2]

    def _calc_pareto_set(self, n_pareto_points=100):
        n = self.n_var
        pareto_set = np.zeros((n_pareto_points, n))
        pareto_set[:, 0] = np.linspace(0.0, 1.0, n_pareto_points)

        for j in range(2, n+1): #TODO: can I write this better?
            frac = (3 * (j-2)) / (n-2)
            pareto_set[:, j-1] = np.power( pareto_set[:, 0], 0.5 * ( 1.0 + frac ) )

        return pareto_set

class LZ2(LZ):
    def __init__(self, n_var=30):
        xl=np.ones(n_var) * -1
        xu=np.ones(n_var)
        xl[0] = 0.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        odd = np.arange(1, self.n_var, 2)
        even = np.arange(2, self.n_var, 2) #zero is not included

        f_1_sum = np.sum( [np.power( x[i] - np.sin(6*np.pi*x[0] + ( (i+1)*np.pi / self.n_var ) ), 2 ) for i in odd] )
        f_2_sum = np.sum( [np.power( x[i] - np.sin(6*np.pi*x[0] + ( (i+1)*np.pi / self.n_var ) ), 2 ) for i in even] )

        f_1 = x[0] + (2 / len(odd)) * f_1_sum
        f_2 = 1 - np.sqrt(x[0]) + (2 / len(even)) * f_2_sum

        out["F"] = [f_1, f_2]

    def _calc_pareto_set(self, n_pareto_points=100):
        n = self.n_var
        pareto_set = np.zeros((n_pareto_points, n))
        pareto_set[:, 0] = np.linspace(self.xl[0], self.xu[0], n_pareto_points)

        for i in range(n_pareto_points):
            for j in range(1, n):
                pareto_set[i, j] = np.sin( 6 * np.pi * pareto_set[i, 0] + ((j+1)* np.pi / self.n_var ) )
        return pareto_set
    
class LZ3(LZ):

    def __init__(self, n_var=30):
        xl=np.ones(n_var) * -1
        xu=np.ones(n_var)
        xl[0] = 0.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        odd = np.arange(1, self.n_var, 2)
        even = np.arange(2, self.n_var, 2) #zero is not included

        f_1_sum = np.sum( [np.power( x[i] - 0.8 * x[0] * np.cos( 6 * np.pi * x[0] + ( (i+1) * np.pi / self.n_var ) ), 2 ) for i in odd] )
        f_2_sum = np.sum( [np.power( x[i] - 0.8 * x[0] * np.sin( 6 * np.pi * x[0] + ( (i+1) * np.pi / self.n_var ) ), 2 ) for i in even] )

        f_1 = x[0] + (2 / len(odd)) * f_1_sum
        f_2 = 1 - np.sqrt(x[0]) + (2 / len(even)) * f_2_sum

        out["F"] = [f_1, f_2]

    def _calc_pareto_set(self, n_pareto_points=100):
        n = self.n_var
        pareto_set = np.zeros((n_pareto_points, n))
        pareto_set[:, 0] = np.linspace(self.xl[0], self.xu[0], n_pareto_points)

        for i in range(n_pareto_points):
            for j in range(1, n):
                if j % 2 == 1: #odd
                    pareto_set[i, j] = 0.8 * pareto_set[i, 0] * np.cos( 6 * np.pi * pareto_set[i, 0] + ((j+1)* np.pi / self.n_var ) )
                else: #even
                    pareto_set[i, j] = 0.8 * pareto_set[i, 0] * np.sin( 6 * np.pi * pareto_set[i, 0] + ((j+1)* np.pi / self.n_var ) )
        return pareto_set
    
class LZ4(LZ):
    def __init__(self, n_var=30):
        xl=np.ones(n_var) * -1
        xu=np.ones(n_var)
        xl[0] = 0.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        odd = np.arange(1, self.n_var, 2)
        even = np.arange(2, self.n_var, 2) #zero is not included

        f_1_sum = np.sum( [np.power( x[i] - 0.8 * x[0] * np.cos( ( 6 * np.pi * x[0] + ((i+1)* np.pi / self.n_var ) ) / 3 ), 2 ) for i in odd] )
        f_2_sum = np.sum( [np.power( x[i] - 0.8 * x[0] * np.sin( ( 6 * np.pi * x[0] + ((i+1)* np.pi / self.n_var ) ) / 3 ), 2 ) for i in even] )

        f_1 = x[0] + (2 / len(odd)) * f_1_sum
        f_2 = 1 - np.sqrt(x[0]) + (2 / len(even)) * f_2_sum

        out["F"] = [f_1, f_2]

    def _calc_pareto_set(self, n_pareto_points=100):
        n = self.n_var
        pareto_set = np.zeros((n_pareto_points, n))
        pareto_set[:, 0] = np.linspace(self.xl[0], self.xu[0], n_pareto_points)

        for i in range(n_pareto_points):
            for j in range(1, n):
                if j % 2 == 1: #odd
                    pareto_set[i, j] = 0.8 * pareto_set[i, 0] * np.cos( ( 6 * np.pi * pareto_set[i, 0] + ((j+1)* np.pi / self.n_var ) ) / 3 )
                else: #even
                    pareto_set[i, j] = 0.8 * pareto_set[i, 0] * np.sin( ( 6 * np.pi * pareto_set[i, 0] + ((j+1)* np.pi / self.n_var ) ) / 3 )
        return pareto_set
    
class LZ5(LZ):
    def __init__(self, n_var=30):
        xl=np.ones(n_var) * -1
        xu=np.ones(n_var)
        xl[0] = 0.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        odd = np.arange(1, self.n_var, 2)
        even = np.arange(2, self.n_var, 2) #zero is not included

        f_1_sum = np.sum( [np.power( x[i] - ( 0.3 * np.power(x[0], 2) * np.cos(24*np.pi*x[0]+(4*(i+1)*np.pi)/self.n_var) + 0.6 * x[0]) * np.cos( 6 * np.pi * x[0] + ((i+1)* np.pi / self.n_var ) ), 2 ) for i in odd] )
        f_2_sum = np.sum( [np.power( x[i] - ( 0.3 * np.power(x[0], 2) * np.cos(24*np.pi*x[0]+(4*(i+1)*np.pi)/self.n_var) + 0.6 * x[0]) * np.sin( 6 * np.pi * x[0] + ((i+1)* np.pi / self.n_var ) ), 2 ) for i in even] )

        f_1 = x[0] + (2 / len(odd)) * f_1_sum
        f_2 = 1 - np.sqrt(x[0]) + (2 / len(even)) * f_2_sum

        out["F"] = [f_1, f_2]

    def _calc_pareto_set(self, n_pareto_points=100):
        n = self.n_var
        pareto_set = np.zeros((n_pareto_points, n))
        pareto_set[:, 0] = np.linspace(self.xl[0], self.xu[0], n_pareto_points)

        for i in range(n_pareto_points):
            for j in range(1, n):
                if j % 2 == 1: #odd
                    pareto_set[i, j] = ( 0.3 * np.power(pareto_set[i, 0], 2) * np.cos(24*np.pi*pareto_set[i, 0]+(4*(j+1)*np.pi)/self.n_var) + 0.6 * pareto_set[i, 0]) * np.cos( 6 * np.pi * pareto_set[i, 0] + ((j+1)* np.pi / self.n_var ) )
                else: #even
                    pareto_set[i, j] = ( 0.3 * np.power(pareto_set[i, 0], 2) * np.cos(24*np.pi*pareto_set[i, 0]+(4*(j+1)*np.pi)/self.n_var) + 0.6 * pareto_set[i, 0]) * np.sin( 6 * np.pi * pareto_set[i, 0] + ((j+1)* np.pi / self.n_var ) )
        return pareto_set
    
class LZ6(LZ):
    def __init__(self, n_var=10):
        xl=np.ones(n_var) * -2
        xu=np.ones(n_var) * 2
        xl[0] = 0.0
        xl[1] = 0.0
        xu[0] = 1.0
        xu[1] = 1.0
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

        f_1 = np.cos(0.5*np.pi*x[0]) * np.cos(0.5*np.pi*x[1]) + (2/len(j_1)) * f_1_sum
        f_2 = np.cos(0.5*np.pi*x[0]) * np.sin(0.5*np.pi*x[1]) + (2/len(j_2)) * f_2_sum
        f_3 = np.sin(0.5*np.pi*x[0]) + (2/len(j_3)) * f_3_sum

        out["F"] = [f_1, f_2, f_3]

    def _calc_pareto_set(self, n_pareto_points=625):
        #calculate the optimal values for x_1 and x_2. Try to get as many equidistant points as possible and calculate the rest randonly
        n_equidistant_samples = np.floor( np.sqrt(n_pareto_points) ).astype(int)
        n_random_samples = n_pareto_points - (n_equidistant_samples * n_equidistant_samples)
        #print(n_equidistant_samples, n_random_samples)
        
        equidistant_samples = np.array( [[x, y] for x in np.linspace(0.0, 1.0, n_equidistant_samples) for y in np.linspace(0.0, 1.0, n_equidistant_samples)] )
        random_samples = np.random.rand(n_random_samples, 2)
        x_1_2 = np.vstack((equidistant_samples, random_samples)) #the optimal values for x[:,0] and x[:,1]

        #set x_1 and x_2
        pareto_set = np.zeros((n_pareto_points, self.n_var))
        pareto_set[:, 0] = x_1_2[:,0]
        pareto_set[:, 1] = x_1_2[:,1]

        #calculate the remaining values
        for j in range(2, self.n_var):
            pareto_set[:, j] = 2*pareto_set[:,1] * np.sin(2*np.pi*pareto_set[:,0] + ( (j+1)*np.pi/self.n_var ) )

        return pareto_set

class LZ7(LZ1): #PS and bounds are the same as LZ1
    def get_y(self, x, i):
        return np.power( x[i] - np.power( x[0], 0.5 * ( 1.0 + ( (3*(i-1) / (self.n_var - 2) ) ) ) ), 2 )
    def _evaluate(self, x, out, *args, **kwargs):
        odd = np.arange(1, self.n_var, 2)
        even = np.arange(2, self.n_var, 2) #zero is not included

        f_1_sum = np.sum( [4*np.power(self.get_y(x,i),2) - np.cos(8*self.get_y(x,i)*np.pi) + 1.0 for i in odd] )
        f_2_sum = np.sum( [4*np.power(self.get_y(x,i),2) - np.cos(8*self.get_y(x,i)*np.pi) + 1.0 for i in even] )

        f_1 = x[0] + (1 / len(odd)) * f_1_sum
        f_2 = 1 - np.sqrt(x[0]) + (1 / len(even)) * f_2_sum
        out["F"] = [f_1, f_2]

class LZ8(LZ7): #PS and bounds are the same as LZ7 (and LZ1)
    def _evaluate(self, x, out, *args, **kwargs):
        odd = np.arange(1, self.n_var, 2)
        even = np.arange(2, self.n_var, 2) #zero is not included

        f_1_sum = np.sum( [np.power(self.get_y(x,i),2) - 2*np.prod( np.cos(20*self.get_y(x,i)*np.pi / np.sqrt(i+1)) ) + 2.0 for i in odd] )
        f_2_sum = np.sum( [np.power(self.get_y(x,i),2) - 2*np.prod( np.cos(20*self.get_y(x,i)*np.pi / np.sqrt(i+1)) ) + 2.0 for i in even] )

        f_1 = x[0] + (1 / len(odd)) * 4 * f_1_sum
        f_2 = 1 - np.sqrt(x[0]) + (1 / len(even)) * 4 * f_2_sum
        out["F"] = [f_1, f_2]

class LZ9(LZ2): #PS and bounds are the same as LZ2

    def _evaluate(self, x, out, *args, **kwargs):
        odd = np.arange(1, self.n_var, 2)
        even = np.arange(2, self.n_var, 2) #zero is not included

        f_1_sum = np.sum( [np.power( x[i] - np.sin(6*np.pi*x[0] + ( (i+1)*np.pi / self.n_var ) ), 2 ) for i in odd] )
        f_2_sum = np.sum( [np.power( x[i] - np.sin(6*np.pi*x[0] + ( (i+1)*np.pi / self.n_var ) ), 2 ) for i in even] )

        f_1 = x[0] + (2 / len(odd)) * f_1_sum
        f_2 = 1 - np.power(x[0], 2) + (2 / len(even)) * f_2_sum #the only difference to LZ2 is here

        out["F"] = [f_1, f_2]