# Pymoo Problems

This repository contains problem implementations for the pymoo [1] framework which I use for my research work. Some of them are just shightly changed pymoo implementations, others are not found in pymoo by default and implemented here.


# Multi-Objective Problems
Here is a short list of the multi-objective problems implemented.

## MACO: Multi-Agent Coordinaation Problem
Reference implementation of the MACO problem as described in [2]. Besides the base-version the three variations (weights, p-norm, interaction classes) are also implemented.

```python
from pymoo_problems.moo.maco import MACO
```

## LZ Problems
The LZ problems 1 - 9 implemented after the reference in [3].

```python
from pymoo_problems.moo.lz import LZ1, LZ2, LZ3, LZ4, LZ5, LZ6, LZ7, LZ8, LZ9
```

## LZ Problems from jmetalpy
A port from the jmetalpy [4] implementation of the LZ problems. The implementation contains references to the PF which are a usefull check. However, the implementation does not stay true to the original LZ problems as described in [3], mainly as they shift the bounds of the problem to $[0.0,1.0]^1 x [0.5,1.0]^{n-1}$ for all problems. This still fits to the original Pareto front, however not the originally described Pareto set. For this reason we implemented our own reference which does fit in the original bounds and the original Pareto set as described in [3].

```python
from pymoo_problems.moo.lz_jmetalpy import LZ09_F1, LZ09_F2, LZ09_F3, LZ09_F4, LZ09_F5, LZ09_F6, LZ09_F7, LZ09_F8, LZ09_F9
```

## CEC 2009 Benchmark Problems
This class is currently just a wrapper for the UF [5] problems which are a direct copy of a LZ problem.

```python
from pymoo_problems.moo.uf import UF1, UF2, UF3, UF8
```

## ZDT Problems
Added a pareto set calculation for ZDT [6] functions ZDT1, ZDT2, ZDT3, ZDT4, and ZDT6 (not ZDT5) to the pymoo [1] implementation.

```python
from pymoo_problems.moo.ZDT import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
```


# References

[1] J. Blank und K. Deb, „Pymoo: Multi-Objective Optimization in Python“, IEEE Access, Bd. 8, S. 89497–89509, 2020, doi: 10.1109/ACCESS.2020.2990567.

[2] S. Mai, T. Benecke, und S. Mostaghim, „MACO: A Real-World Inspired Benchmark for Multi-objective Evolutionary Algorithms“, in Evolutionary Multi-Criterion Optimization, in Lecture Notes in Computer Science. Cham: Springer Nature Switzerland, 2023, S. 305–318. doi: 10.1007/978-3-031-27250-9_22.

[3] H. Li und Q. Zhang, „Multiobjective Optimization Problems With Complicated Pareto Sets, MOEA/D and NSGA-II“, IEEE Transactions on Evolutionary Computation, Bd. 13, Nr. 2, S. 284–302, Apr. 2009, doi: 10.1109/TEVC.2008.925798.

[4] A. Benítez-Hidalgo, A. J. Nebro, J. García-Nieto, I. Oregi, und J. Del Ser, „jMetalPy: A Python framework for multi-objective optimization with metaheuristics“, Swarm and Evolutionary Computation, Bd. 51, S. 100598, Dez. 2019, doi: 10.1016/j.swevo.2019.100598.

[5] Q. Zhang, A. Zhou, S. Zhao, P. Suganthan, W. Liu, und S. Tiwari, „Multiobjective optimization Test Instances for the CEC 2009 Special Session and Competition“, Mechanical Engineering, Jan. 2008.

[6] E. Zitzler, K. Deb, und L. Thiele, „Comparison of Multiobjective Evolutionary Algorithms: Empirical Results“, Evol. Comput., Bd. 8, Nr. 2, S. 173–195, Juni 2000, doi: 10.1162/106365600568202.

