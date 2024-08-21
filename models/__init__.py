# Load the benchmarks from the subfiles
from .collision_avoidance import CollisionAvoidance
from .linearsystem import LinearSystem
from .pendulum import Pendulum
from .triple_integrator import TripleIntegrator


def get_model_fun(model_name):
    if model_name == 'LinearSystem':
        envfun = LinearSystem
    elif model_name == 'MyPendulum':
        envfun = Pendulum
    elif model_name == 'CollisionAvoidance':
        envfun = CollisionAvoidance
    elif model_name == 'TripleIntegrator':
        envfun = TripleIntegrator
    else:
        envfun = False
        assert False, f"Unknown model name: {model_name}"

    return envfun
