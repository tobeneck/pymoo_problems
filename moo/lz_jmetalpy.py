import math
import numpy as np

from pymoo.core.problem import Problem

"""
Direct port of the jmetalpy implementation of the LZ09 problem family.
"""


class LZ09(Problem):

    def __init__(
        self,
        xl,
        xu,
        n_var: int,
        n_obj: int,
        ptype: int,
        dtype: int,
        ltype: int,
    ):
        """LZ09 benchmark family as defined in:

        * H. Li and Q. Zhang. Multiobjective optimization problems with complicated pareto sets, MOEA/D and NSGA-II.
        IEEE Transactions on Evolutionary Computation, 12(2):284-302, April 2009.
        """
        super().__init__(n_var=n_var,
                             n_obj=n_obj,
                             xl=xl,
                             xu=xu,
                             elementwise=True,
                             )

        self.ptype = ptype
        self.dtype = dtype
        self.ltype = ltype
    
    def _evaluate(self, x, out, *args, **kwargs):
        fitness = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            fitness[i] = self.objective(x)[i]
        out["F"] = fitness

    def __ps_func2(self, x: float, t1: float, dim: int, type: int, css: int) -> float:
        """Control the PS shapes of 2-D instances.

        :param type: The type of the curve.
        :param css: The class of the index.
        """
        beta = 0.0
        dim += 1

        if type == 21:
            xy = x#2 * (x - 0.5) #TODO: change back!
            beta = xy - math.pow(t1, 0.5 * (self.n_var + 3 * dim - 8) / (self.n_var - 2))
        if type == 22:
            theta = 6 * math.pi * t1 + dim * math.pi / self.n_var
            xy = x#2 * (x - 0.5)#TODO: change back!
            beta = xy - math.sin(theta)
        if type == 23:
            theta = 6 * math.pi * t1 + dim * math.pi / self.n_var
            ra = 0.8 * t1
            xy = x#2 * (x - 0.5)#TODO: change back!
            if css == 1:
                beta = xy - ra * math.cos(theta)
            else:
                beta = xy - ra * math.sin(theta)
        if type == 24:
            theta = 6 * math.pi * t1 + dim * math.pi / self.n_var
            xy = x#2 * (x - 0.5)#TODO: change back!
            ra = 0.8 * t1
            if css == 1:
                beta = xy - ra * math.cos(theta / 3)
            else:
                beta = xy - ra * math.sin(theta)
        if type == 25:
            rho = 0.8
            phi = math.pi * t1
            theta = 6 * math.pi * t1 + dim * math.pi / self.n_var
            xy = x#2 * (x - 0.5)#TODO: change back!
            if css == 1:
                beta = xy - rho * math.sin(phi) * math.sin(theta)
            elif css == 2:
                beta = xy - rho * math.sin(phi) * math.cos(theta)
            else:
                beta = xy - rho * math.cos(phi)
        if type == 26:
            theta = 6 * math.pi * t1 + dim * math.pi / self.n_var
            ra = 0.3 * t1 * (t1 * math.cos(4 * theta) + 2)
            xy = x#2 * (x - 0.5)#TODO: change back!
            if css == 1:
                beta = xy - ra * math.cos(theta)
            else:
                beta = xy - ra * math.sin(theta)

        return beta

    def __ps_func3(self, x: float, t1: float, t2: float, dim: int, type: int):
        """Control the PS shapes of 3-D instances.
        :param type: The type of curve.
        """
        beta = 0.0
        dim += 1

        if type == 31:
            xy = x#4 * (x - 0.5)
            rate = 1.0 * dim / self.n_var
            beta = xy - 4 * (t1 * t1 * rate + t2 * (1.0 - rate)) + 2
        if type == 32:
            theta = 2 * math.pi * t1 + dim * math.pi / self.n_var
            xy = x#4 * (x - 0.5)
            beta = xy - 2 * t2 * math.sin(theta)

        return beta

    def __alpha_func(self, x: list, dim: int, type: int) -> list:
        """Control the PF shape."""
        alpha = [0.0] * dim

        if dim == 2:
            if type == 21:
                alpha[0] = x[0]
                alpha[1] = 1 - math.sqrt(x[0])
            if type == 22:
                alpha[0] = x[0]
                alpha[1] = 1 - x[0] * x[0]
            if type == 23:
                alpha[0] = x[0]
                alpha[1] = 1 - math.sqrt(alpha[0]) - alpha[0] * math.sin(10 * alpha[0] * alpha[0] * math.pi)
            if type == 24:
                alpha[0] = x[0]
                alpha[1] = 1 - x[0] - 0.05 * math.sin(4 * math.pi * x[0])
        else:
            if type == 31:
                alpha[0] = math.cos(x[0] * math.pi / 2) * math.cos(x[1] * math.pi / 2)
                alpha[1] = math.cos(x[0] * math.pi / 2) * math.sin(x[1] * math.pi / 2)
                alpha[2] = math.sin(x[0] * math.pi / 2)
            if type == 32:
                alpha[0] = 1 - math.cos(x[0] * math.pi / 2) * math.cos(x[1] * math.pi / 2)
                alpha[1] = 1 - math.cos(x[0] * math.pi / 2) * math.sin(x[1] * math.pi / 2)
                alpha[2] = 1 - math.sin(x[0] * math.pi / 2)
            if type == 33:
                alpha[0] = x[0]
                alpha[1] = x[1]
                alpha[2] = 3 - (math.sin(3 * math.pi * x[0]) + math.sin(3 * math.pi * x[1]) - 2 * (x[0] + x[1]))
            if type == 34:
                alpha[0] = x[0] - x[1]
                alpha[1] = x[0] * (1 - x[1])
                alpha[2] = 1 - x[0]

        return alpha

    def __beta_func(self, x: list, type: int) -> float:
        """Control the distance."""
        beta = 0.0
        dim = len(x)

        if dim == 0:
            beta = 0.0
        if type == 1:
            for i in range(dim):
                beta += x[i] * x[i]
            beta = 2.0 * beta / dim
        if type == 2:
            for i in range(dim):
                beta += math.sqrt(i + 1) * x[i] * x[i]
            beta = 2.0 * beta / dim
        if type == 3:
            sum, xx = 0, 0
            for i in range(dim):
                xx = 2 * x[i]
                sum += xx * xx - math.cos(4 * math.pi * xx) + 1
            beta = 2.0 * sum / dim
        if type == 4:
            sum, prod, xx = 0, 1, 0
            for i in range(dim):
                xx = 2 * x[i]
                sum += xx * xx
                prod *= math.cos(10 * math.pi * xx / math.sqrt(i + 1))
            beta = 2.0 * (sum - 2 * prod + 2) / dim

        return beta

    def objective(self, x_variables: np.array) -> list:
        aa = []
        bb = []
        cc = []

        y_objectives = [0.0] * self.n_obj

        if self.n_obj == 2:
            if self.ltype in [21, 22, 23, 24, 26]:
                for n in range(1, self.n_var):
                    if n % 2 == 0:
                        a = self.__ps_func2(x_variables[n], x_variables[0], n, self.ltype, 1)
                        aa.append(a)
                    else:
                        b = self.__ps_func2(x_variables[n], x_variables[0], n, self.ltype, 2)
                        bb.append(b)

                g = self.__beta_func(aa, self.dtype)
                h = self.__beta_func(bb, self.dtype)

                alpha = self.__alpha_func(x_variables, 2, self.ptype)

                y_objectives[0] = alpha[0] + h
                y_objectives[1] = alpha[1] + g
            if self.ltype == 25:
                for n in range(1, self.n_var):
                    if n % 3 == 0:
                        a = self.__ps_func2(x_variables[n], x_variables[0], n, self.ltype, 1)
                        aa.append(a)
                    elif n % 3 == 1:
                        b = self.__ps_func2(x_variables[n], x_variables[n], n, self.ltype, 2)
                        bb.append(b)
                    else:
                        c = self.__ps_func2(x_variables[n], x_variables[0], n, self.ltype, 3)
                        if n % 2 == 0:
                            aa.append(c)
                        else:
                            bb.append(c)

                g = self.__beta_func(aa, self.dtype)
                h = self.__beta_func(bb, self.dtype)

                alpha = self.__alpha_func(x_variables, 2, self.ptype)

                y_objectives[0] = alpha[0] + h
                y_objectives[1] = alpha[1] + g

        if self.n_obj == 3:
            if self.ltype == 31 or self.ltype == 32:

                for n in range(2, self.n_var):
                    a = self.__ps_func3(x_variables[n], x_variables[0], x_variables[1], n, self.ltype)

                    if n % 3 == 0:
                        aa.append(a)
                    elif n % 3 == 1:
                        bb.append(a)
                    else:
                        cc.append(a)

                g = self.__beta_func(aa, self.dtype)
                h = self.__beta_func(bb, self.dtype)
                e = self.__beta_func(cc, self.dtype)

                alpha = self.__alpha_func(x_variables, 3, self.ptype)

                y_objectives[0] = alpha[0] + h
                y_objectives[1] = alpha[1] + g
                y_objectives[2] = alpha[2] + e

        return y_objectives

    def _calc_pareto_front(self, n_pareto_points=100):
        '''
        Returns a sample of the pareto front.

        Note that this just reads the pre-computed pareto front available from the jemtalpy implementation. No points are generated, the parameter n_pareto_points is ignored.

        Parameters:
        -----------
        n_pareto_points: int
            The number of points to sample. Will be ignored in this case.

        Returns:
        --------
        pareto_front: np.array
            500 sample points from the pareto front.
        '''
        pareto_front = np.genfromtxt(self.pf_path, delimiter=",")
        return pareto_front


import math
class LZ09_F1(LZ09):
    def __init__(self, n_var=10):
        xl=0.0
        xu=1.0
        super().__init__(
            n_var=n_var, 
            xl=xl,
            xu=xu,
            n_obj=2,
            dtype=1,
            ltype=21,
            ptype=21
        )
        self.pf_path = "./moo/pareto_fronts/LZ09_F1.csv"

    def _calc_pareto_set(self, n_pareto_points=100):
        n = self.n_var
        pareto_set = np.zeros((n_pareto_points, n))
        pareto_set[:, 0] = np.linspace(0.0, 1.0, n_pareto_points)

        for j in range(2, n+1): #TODO: can I write this better?
            frac = (3 * (j-2)) / (n-2)
            pareto_set[:, j-1] = np.power( pareto_set[:, 0], 0.5 * ( 1.0 + frac ) )

        return pareto_set

class my_LZ09_F1(LZ09_F1):
    def test_eval_ind(self, x):
        '''test evaluate one individual.'''
        f_1_sum = 0.0
        f_2_sum = 0.0

        n_even = 0
        n_odd = 0

        for i in range(2, self.n_var+1):
            x_1 = math.pow( x[0], 0.5 * ( 1.0 + ( (3*(i-2) / (self.n_var - 2) ) ) ) )
            if i % 2 == 0:
                n_even+=1
                f_2_sum += math.pow( (x[i-1] - x_1), 2)
            else:
                n_odd+=1
                f_1_sum += math.pow( (x[i-1] - x_1), 2)
                

        f_1 = x[0] + (1 / n_odd) * f_1_sum
        f_2 = 1 - math.sqrt(x[0]) + (1 / n_odd) * f_2_sum
        return [f_1, f_2]
       
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.test_eval_ind(x)

class LZ09_F2(LZ09):
    def __init__(self, n_var=30):
        xl=np.ones(n_var) * -1
        xu=np.ones(n_var)
        xl[0] = 0.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
            dtype=1,
            ltype=22,
            ptype=21
        )
        self.pf_path = "./moo/pareto_fronts/LZ09_F2.csv"
    
    def _calc_pareto_set(self, n_pareto_points=100):
        n = self.n_var
        pareto_set = np.zeros((n_pareto_points, n))
        pareto_set[:, 0] = np.linspace(self.xl[0], self.xu[0], n_pareto_points)

        # for j in range(2, n+1): #TODO: can I write this better?
        #     pareto_set[:, j-1] = np.sin( 6 * np.pi * pareto_set[:, 0] + ( n * np.pi / self.n_var ) )
        for i in range(n_pareto_points):
            for j in range(1, n):
                pareto_set[i, j] = math.sin( 6 * math.pi * pareto_set[i, 0] + ( j+1 * math.pi / self.n_var ) )
        return pareto_set

class my_LZ09_F2(LZ09_F2):
    def _evaluate(self, x, out, *args, **kwargs):
        odd = np.arange(1, self.n_var, 2)
        even = np.arange(2, self.n_var, 2) #zero is not included

        f_1_sum = sum( [math.pow( x[i] - math.sin( 6 * math.pi * x[0] + ( i+1 * math.pi / self.n_var ) ), 2 ) for i in odd] )
        f_2_sum = sum( [math.pow( x[i] - math.sin( 6 * math.pi * x[0] + ( i+1 * math.pi / self.n_var ) ), 2 ) for i in even] )

        f_1 = x[0] + (2 / len(odd)) * f_1_sum
        f_2 = 1 - math.sqrt(x[0]) + (2 / len(even)) * f_2_sum

        out["F"] = [f_1, f_2]
        
class LZ09_F3(LZ09):
    def __init__(self, n_var=30):
        xl=np.ones(n_var) * -1
        xu=np.ones(n_var)
        xl[0] = 0.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
            dtype=1,
            ltype=23,
            ptype=21
        )
        self.pf_path = "./moo/pareto_fronts/LZ09_F3.csv"


class LZ09_F4(LZ09):
    def __init__(self, n_var=30):
        xl=np.ones(n_var) * -1
        xu=np.ones(n_var)
        xl[0] = 0.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
            dtype=1,
            ltype=24,
            ptype=21
        )
        self.pf_path = "./moo/pareto_fronts/LZ09_F4.csv"

class LZ09_F5(LZ09):
    def __init__(self, n_var=30):
        xl=np.ones(n_var) * -1
        xu=np.ones(n_var)
        xl[0] = 0.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
            dtype=1,
            ltype=26,
            ptype=21
        )
        self.pf_path = "./moo/pareto_fronts/LZ09_F5.csv"


class LZ09_F6(LZ09):
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
            dtype=1,
            ltype=32,
            ptype=31
        )
        self.pf_path = "./moo/pareto_fronts/LZ09_F6.csv"
    
    # def _calc_pareto_set(self, n_pareto_points=100):
    #     n = self.n_var

    #     x1s = np.linspace(self.xl[0], self.xu[0], 10) #TODO: 10 * 10 = 100 points. Can I make this scalable?
    #     x2s = np.linspace(self.xl[1], self.xu[1], 10)

    #     pareto_set = np.zeros((n_pareto_points, n))

    #     for x1 in x1s:
    #         for x2 in x2s:
                

    #     pareto_set = np.zeros((n_pareto_points, n))
    #     pareto_set[:, 0] = np.linspace(self.xl[0], self.xu[0], n_pareto_points)

    #     # for j in range(2, n+1): #TODO: can I write this better?
    #     #     pareto_set[:, j-1] = np.sin( 6 * np.pi * pareto_set[:, 0] + ( n * np.pi / self.n_var ) )
    #     for i in range(n_pareto_points):
    #         for j in range(1, n):
    #             pareto_set[i, j] = math.sin( 6 * math.pi * pareto_set[i, 0] + ( j * math.pi / self.n_var ) )
    #     return pareto_set


class LZ09_F7(LZ09):
    def __init__(self, n_var=10):
        xl=0.0
        xu=1.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
            dtype=3,
            ltype=21,
            ptype=21
        )
        self.pf_path = "./moo/pareto_fronts/LZ09_F7.csv"
    
    def _calc_pareto_set(self, n_pareto_points=100):
        LZ01 = LZ09_F1(n_var=self.n_var) #has the same PS as F1
        return LZ01._calc_pareto_set(n_pareto_points=n_pareto_points)


class LZ09_F8(LZ09):
    def __init__(self, n_var=10):
        xl=0.0
        xu=1.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
            dtype=4,
            ltype=21,
            ptype=21
        )
        self.pf_path = "./moo/pareto_fronts/LZ09_F8.csv"
    
    def _calc_pareto_set(self, n_pareto_points=100):
        LZ01 = LZ09_F1(n_var=self.n_var) #has the same PS as F1
        return LZ01._calc_pareto_set(n_pareto_points=n_pareto_points)


class LZ09_F9(LZ09):
    def __init__(self, n_var=30):
        xl=np.ones(n_var) * -1
        xu=np.ones(n_var)
        xl[0] = 0.0
        super().__init__(
            n_var=n_var,
            xl=xl,
            xu=xu,
            n_obj=2,
            dtype=1,
            ltype=22,
            ptype=22
        )
        self.pf_path = "./moo/pareto_fronts/LZ09_F9.csv"
    
    def _calc_pareto_set(self, n_pareto_points=100):
        LZ02 = LZ09_F2(n_var=self.n_var) #has the same PS as F2
        return LZ02._calc_pareto_set(n_pareto_points=n_pareto_points)
    
