import numpy as np
from pymoo.core.problem import Problem

from lz import LZ2, LZ5, LZ6, LZ8

"""
Implementation of the LZ problem family as defined in:
Q. Zhang, A. Zhou, S. Zhao, P. Suganthan, W. Liu, und S. Tiwari, „Multiobjective optimization Test Instances for the CEC 2009 Special Session and Competition“, Mechanical Engineering, Jan. 2008.

Currently just wrapping the corresponding Z problems.
"""

class UF1(LZ2): #exatly the same as LZ2
    pass

class UF2(LZ5): #exatly the same as LZ5
    pass

class UF3(LZ8): #exatly the same as LZ6
    pass

class UF8(LZ6): #exatly the same as LZ6
    pass