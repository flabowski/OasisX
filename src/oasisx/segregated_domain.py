__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-09"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

import numpy as np
import matplotlib.pyplot as plt
from oasisx.domain import Domain
from oasisx.logging import info_blue
import dolfin as df
from ufl import Coefficient
from enum import Enum
from dataclasses import dataclass, field


@dataclass()
class Solver:
    method: str = "WeightedAverage"  # default
    solver_type: str = "cg"  # gmres, bicgstab
    preconditioner_type: str = "jacobi"  # hypre_amg
    low_memory_version: bool = True


@dataclass()
class PETScKrylovSolverParameters:
    absolute_tolerance: float = 1e-08
    # convergence_norm_type: NoneType = None
    # divergence_limit: NoneType = None
    error_on_nonconvergence: bool = False
    maximum_iterations: int = 200
    monitor_convergence: bool = False
    nonzero_initial_guess: bool = True
    relative_tolerance: float = 1e-8
    report: bool = False


# only IPCS_ABCN is implemented yet. Its thus used blindly by default.
class Scheme(Enum):
    IPCS_ABCN = 1
    IPCS_ABE = 2
    IPCS = 3
    Chorin = 4
    BDFPC = 5
    BDFPC_Fast = 6


class SegregatedDomain(Domain):
    def __init__(self):
        # super.__init__()
        # self.constrained_domain = None
        # self.velocity_degree = 2
        # self.pressure_degree = 1
        # TODO: move to solver interface.
        # solver specific things should not be here.
        # why not initiate solvers directly?
        # vks = velocity_krylov_solver
        # u_prec = PETScPreconditioner(vks["preconditioner_type"])
        # u_sol = PETScKrylovSolver(vks["solver_type"], u_prec)
        # u_sol.parameters.update(dmn.krylov_solvers)
        return

    def list_problem_components(self):
        # Create lists of components solved for
        # self.scalar_components = scalar_components
        if self.mesh.geometry().dim() == 1:
            self.u_components = ["u0"]
        elif self.mesh.geometry().dim() == 2:
            self.u_components = ["u0", "u1"]
        elif self.mesh.geometry().dim() == 3:
            self.u_components = ["u0", "u1", "u2"]
        self.sys_comp = self.u_components + ["p"] + self.scalar_components
        self.uc_comp = self.u_components + self.scalar_components
        # sys_comp = ['u0', 'u1', 'p', 'alfa']
        # u_components = ['u0', 'u1']
        # uc_comp = ['u0', 'u1', 'alfa']
        return

    def declare_coefficients(self):
        """overload this in case of variable coefficients. E.g.:
        V, Q = self.VV["t"], self.VV["p"]
        self.mu, self.nu, self.rho = Function(V), Function(V), Function(V)
        """
        return

    def declare_components(self):
        cd = self.config["constrained_domain"]
        mesh = self.mesh
        sys_comp = self.sys_comp
        deg_v = self.config["velocity_degree"]
        deg_p = self.config["pressure_degree"]
        V = Q = df.FunctionSpace(mesh, "CG", deg_v, constrained_domain=cd)
        if deg_v != deg_p:
            Q = df.FunctionSpace(mesh, "CG", deg_p, constrained_domain=cd)
        # self.V, self.Q = V, Q
        # R = df.FunctionSpace(mesh, "DG", 0)
        # self.R = R
        self.u, self.v = df.TrialFunction(V), df.TestFunction(V)
        self.p, self.q = df.TrialFunction(Q), df.TestFunction(Q)

        # Use dictionary to hold all FunctionSpaces
        VV = dict((ui, V) for ui in self.uc_comp)
        VV["p"] = Q
        self.VV = VV

        # removed unused name argument and reassigning q_[...].vector() to x_..
        # Create dictionaries for the solutions at three timesteps
        self.q_ = dict((ui, df.Function(VV[ui])) for ui in sys_comp)
        self.q_1 = dict((ui, df.Function(VV[ui])) for ui in sys_comp)
        self.q_2 = dict((ui, df.Function(V)) for ui in self.u_components)
        # Create vectors of the segregated velocity components
        self.u_ = df.as_vector([self.q_[ui] for ui in self.u_components])
        self.u_1 = df.as_vector([self.q_1[ui] for ui in self.u_components])
        self.u_2 = df.as_vector([self.q_2[ui] for ui in self.u_components])
        # Adams Bashforth projection of velocity at t - dt/2
        self.U_AB = 1.5 * self.u_1 - 0.5 * self.u_2
        # Create vectors to hold rhs of equations
        self.b = dict((ui, df.Vector(self.q_[ui].vector())) for ui in sys_comp)
        self.b_tmp = dict(
            (ui, df.Vector(self.q_[ui].vector())) for ui in sys_comp
        )
        self.dp_ = df.Function(Q)  # pressure correction

        # TODO: remove u_, u_1, u_2 -> redundand!
        # x_, x_1, x_2 removed, they are in q_[...].vector(), q_1[...].vector(), q_2[...].vector()
        # alpha_, alpha_1 removed, they are in q_ and q_1
        # p_, p_1 removed, they are in q_, q_1
        self.declare_coefficients()  # in case there are any..
        self.b0 = b0 = {}  # holds body forces
        for i, ui in enumerate(self.u_components):
            self.b0[ui] = df.assemble(self.v * df.dx)

        # Get scalar sources
        self.fs = fs = self.scalar_source()
        for ci in self.scalar_components:
            assert isinstance(fs[ci], Coefficient)
            b0[ci] = df.assemble(self.v * fs[ci] * df.dx)
        return

    def initialize_components(self):
        return

    def apply_bcs(self):
        # used to be initialize(x_1, x_2, bcs, **NS_namespace)
        for ui in self.sys_comp:
            [bc.apply(self.q_1[ui].vector()) for bc in self.bcs[ui]]
        for ui in self.u_components:
            [bc.apply(self.q_2[ui].vector()) for bc in self.bcs[ui]]
        return

    def advance(self):
        print("ADVANCING!")
        # Update to a new timestep
        # replaced axpy with assign
        for ui in self.u_components:
            self.q_2[ui].assign(self.q_1[ui])
            self.q_1[ui].assign(self.q_[ui])
            # self.q_2[ui].vector().zero()
            # self.q_2[ui].vector().axpy(1.0, self.q_1[ui].vector())
            # self.q_1[ui].vector().zero()
            # self.q_1[ui].vector().axpy(1.0, self.q_[ui].vector())

        for ci in self.scalar_components:
            self.q_1[ci].assign(self.q_[ci])
            # self.q_1[ci].vector().zero()
            # self.q_1[ci].vector().axpy(1.0, self.q_[ci].vector())
        return

    # def initial_hook(self, **kvargs):
    #     dx, dy = V.tabulate_dof_coordinates().T  # 10942,2
    #     # dofmap = V.dofmap()  # len(dofmap.dofs()) = 1042
    #     msh = V.mesh()
    #     mx, my = msh.coordinates().T  # 2805, 2
    #     cells = msh.cells()
    #     fig, ax = plt.subplots()
    #     # ax.plot(mx, my, "ro")
    #     plt.triplot(mx, my, cells, "o")
    #     ax.plot(dx, dy, "k.")
    #     ax.set_aspect("equal")

    def velocity_tentative_hook(self, **kvargs):
        """Called just prior to solving for tentative velocity."""
        pass

    def pressure_hook(self, **kvargs):
        """Called prior to pressure solve."""
        pass

    def start_timestep_hook(self, **kvargs):
        """Called at start of new timestep"""
        pass

    def temporal_hook(self, **kvargs):
        """Called at end of a timestep."""
        pass

    def print_velocity_pressure_info(self, num_iter, inner_iter, udiff):
        cond = self.config["print_velocity_pressure_convergence"]
        if num_iter > 1 and cond:
            if inner_iter == 1:
                info_blue("  Inner iterations velocity pressure:")
                info_blue("                 error u  error p")
            info_blue(
                "    Iter = {0:4d}, {1:2.2e} {2:2.2e}".format(
                    inner_iter, udiff, df.norm(self.dp_.vector())
                )
            )


if __name__ == "__main__":
    # print()
    seg_domain = SegregatedDomain()

    # import json

    # with open("./solver_defaults.json", "r") as infile:
    #     defaults = json.load(infile)
    # seg_domain.__dict__ = defaults
    # with open("./defaults.json", "x") as outfile:
    #     json.dump(seg_domain.__dict__, outfile, indent=2)
