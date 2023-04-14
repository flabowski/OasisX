import pytest
import inspect
import json
from oasisx.fractional_step import FractionalStepAlgorithm
from oasisx.segregated_domain import SegregatedDomain
from oasisx.freezing_cavity import FreezingCavity
import oasisx.ipcs_abcn as solver
from oasisx.io import mesh_from_file

# remember: red, green, refactor. commit frequently.


def test_hello_pytest():
    print("hi pytest")
    return


class Whatever:
    istwo = 2
    isthree = 3


@pytest.mark.parametrize(
    "test_input, expected", [("1", Whatever.istwo), (2, Whatever.isthree)]
)
def test_parameterize(test_input, expected):
    assert int(test_input) + 1 is expected
    return


def test_freezing_cavity():
    pkg_dir = inspect.getfile(SegregatedDomain).split("src")[0]
    path_to_config = pkg_dir + "/resources/freezing_cavity/"
    # path_to_config = pkg_dir + "/checkpoints/20230331_171543/"

    mesh_name = path_to_config + "/mesh.xdmf"
    facet_name = path_to_config + "/mf.xdmf"
    config_file = path_to_config + "config.json"

    print(config_file)
    with open(config_file, encoding="utf-8") as infile:
        domain_config = json.load(infile)
    # commandline_args = parse_command_line()
    with open("../src/oasisx/solver_defaults.json", "r") as infile:
        solver_config = json.load(infile)
    mesh = mesh_from_file(mesh_name, facet_name)
    my_domain = FreezingCavity(domain_config, mesh)
    # my_domain.set_parameters(commandline_args)
    # my_domain.mesh_from_file()
    print("restart", my_domain.restart)

    fit = solver.FirstInner(my_domain)
    tvs = solver.TentativeVelocityStep(my_domain, solver_config)
    prs = solver.PressureStep(my_domain, solver_config)
    for ci in my_domain.scalar_components:
        scs = solver.ScalarSolver(my_domain, solver_config)

    algorithm = FractionalStepAlgorithm(
        my_domain, fit, tvs, prs, scs, solver_config
    )
    # if my_domain.restart:

    my_domain.plot()
    algorithm.run()


if __name__ == "__main__":
    test_freezing_cavity()
