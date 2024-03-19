from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm
from pymoo.core.result import Result

import plotly.graph_objects as go

def run_test(problem:Problem,
            algorithm:Algorithm=NSGA2(pop_size=200),
            n_gen=200
            ):
    '''
    Runs one test and returns the res object of the minimize function for evaluation.

    Standart settings of NSGA-II are used with a pop size of 200 and 200 generations.

    Parameters:
    -----------
    problem: Problem
        The problem to be tested.
    algorithm: Algorithm
        The algorithm to be used for the test.
    
    Returns:
    --------
    res: minimize
        The result of the test.
    '''
    return minimize(problem,
                algorithm,
                ('n_gen', n_gen),
                seed=1,
                verbose=False)


def plot_front(problem:Problem, nds=None, title:str="") -> go.Figure:
    '''
    Plots the pareto front of a problem.

    Parameters:
    -----------
    problem: pymoo.Problem
        The problem the data was generated with.
    nds: np.array | None
        The reulting non-dominated solutions from a test. If None, only the pareto front is plotted.
    '''
    if problem.n_obj == 2:
        return plot_2d_front(problem, nds, title)
    elif problem.n_obj == 3:
        return plot_3d_front(problem, nds, title)
    else:
        print("Can't plot front for problems with more than 3 objectives.")

def plot_2d_front(problem:Problem, nds=None, title:str="") -> go.Figure:
    '''
    Plots the pareto front of a 2D problem.

    Parameters:
    -----------
    problem: pymoo.Problem
        The problem the data was generated with.
    nds: np.array | None
        The reulting non-dominated solutions from a test. If None, only the pareto front is plotted.
    '''
    fig = go.Figure()

    pareto_front = problem.pareto_front()

    # Add traces
    fig.add_trace(go.Scatter(x=pareto_front[:,0], y=pareto_front[:,1],
                        mode='lines',
                        name='PF'))
    if nds is not None:
        fig.add_trace(go.Scatter(x=nds[:,0], y=nds[:,1],
                            mode='markers',
                            name='NDS'))
    
    fig.update_layout(
        autosize=False,
        margin=dict(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=40,  # top margin
            pad=0  # padding
        ),
        height=400,
        width=500,
        title=title,
        font_size=12,
    )

    return fig

def plot_3d_front(problem:Problem, nds=None, title:str="") -> go.Figure:
    '''
    Plots the pareto front of a 3D problem.

    Parameters:
    -----------
    problem: pymoo.Problem
        The problem the data was generated with.
    nds: np.array | None
        The reulting non-dominated solutions from a test. If None, only the pareto front is plotted.
    '''
    fig = go.Figure()

    pareto_front = problem.pareto_front()

    # Add traces
    fig = go.Figure(data=[go.Mesh3d(
                   x=(pareto_front[:,0]),
                   y=(pareto_front[:,1]),
                   z=(pareto_front[:,2]),
                   opacity=0.5,
                   color='rgba(244,22,100,0.6)'
                  )])
    if nds is not None:
        fig.add_trace(go.Scatter3d(x=nds[:,0], y=nds[:,1],z=nds[:,2],
                        mode='markers',
                        marker=dict(
                            size=5,
                            opacity=0.9,
                        ),
                        name='NDS'))
    
    fig.update_layout(
        autosize=True,
        margin=dict(
            l=0,  # left margin
            r=0,  # right margin
            b=40,  # bottom margin
            t=40,  # top margin
            pad=0  # padding
        ),
        height=500,
        width=600,
        title=title,
        font_size=12,
    )
    return fig