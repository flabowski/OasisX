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


class Scheme(Enum):
    IPCS_ABCN = 1
    IPCS_ABE = 2
    IPCS = 3
    Chorin = 4
    BDFPC = 5
    BDFPC_Fast = 6


class SegregatedDomain(Domain):
    def __init__(self):
        # self.nu = 0.01  # Kinematic viscosity
        self.folder = "results"  # Relative folder for storing results
        self.velocity_degree = 2  # default velocity degree
        self.pressure_degree = 1  # default pressure degree

        # Physical constants and solver parameters
        # self.t = 0.0  # Time
        # self.tstep = 0  # Timestep
        # self.T = 1.0  # End time
        # self.dt = 1 / 1600  # 0.01  # Time interval on each timestep

        # Some discretization options
        # Use Adams Bashforth projection as first estimate for pressure on new timestep
        self.AB_projection_pressure = False
        # "IPCS_ABCN", "IPCS_ABE", "IPCS", "Chorin", "BDFPC", "BDFPC_Fast"
        self.solver = Scheme.IPCS_ABCN

        # Parameters used to tweek solver
        # Number of inner pressure velocity iterations on timestep
        # self.max_iter = 5  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Tolerance for inner iterations pressure velocity iterations
        # self.max_error = 1e-6  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Number of iterations on first timestep
        self.iters_on_first_timestep = 5
        self.use_krylov_solvers = True  # Otherwise use LU-solver
        self.print_intermediate_info = 100
        self.print_velocity_pressure_convergence = False

        # Parameters used to tweek output
        self.plot_interval = 100
        # Overwrite solution in Checkpoint folder each checkpoint
        self.checkpoint = 50
        self.save_step = 10  # Store solution each save_step
        # If restarting solution, set the folder holding the solution to start from here
        self.restart_folder = None
        # Store velocity as vector in Timeseries
        self.output_timeseries_as_vector = True
        # Stop simulations cleanly after the given number of seconds
        self.killtime = None

        # Choose LES model and set default parameters
        # NoModel, Smagorinsky, Wale, DynamicLagrangian, ScaleDepDynamicLagrangian
        self.les_model = "NoModel"
        # LES model parameters
        self.Smagorinsky = {"Cs": 0.1677}  # Standard Cs, same as OpenFOAM
        self.Wale = {"Cw": 0.325}
        # Time step interval for Cs to be recomputed
        self.DynamicSmagorinsky = {"Cs_comp_step": 1}
        self.KineticEnergySGS = {"Ck": 0.08, "Ce": 1.05}

        # Choose Non-Newtonian model and set default parameters
        # NoModel, ModifiedCross
        self.nn_model = "NoModel"
        # Non-Newtonian model parameters
        self.ModifiedCross = {
            "lam": 3.736,  # s
            "m_param": 2.406,  # for Non-Newtonian model
            "a_param": 0.34,  # for Non-Newtonian model
            "mu_inf": 0.00372,  # Pa-s for non-Newtonian model
            "mu_o": 0.09,  # Pa-s for non-Newtonian model
            "rho": 1085,  # kg/m^3
        }
        # Parameter set when enabling test mode
        self.testing = False
        # Solver parameters that will be transferred to dolfins parameters['krylov_solver']
        self.krylov_solvers = {
            "monitor_convergence": False,
            "report": False,
            "error_on_nonconvergence": False,
            "nonzero_initial_guess": True,
            "maximum_iterations": 200,
            "relative_tolerance": 1e-8,
            "absolute_tolerance": 1e-8,
        }
        # Velocity update
        self.velocity_update_solver = {
            "method": "default",  # "lumping", "gradient_matrix"
            "solver_type": "cg",
            "preconditioner_type": "jacobi",
            "low_memory_version": False,
        }
        #
        # TODO: move to solver interface.
        # solver specific things should not be here.
        # why not initiate solvers directly?
        # vks = velocity_krylov_solver
        # u_prec = PETScPreconditioner(vks["preconditioner_type"])
        # u_sol = PETScKrylovSolver(vks["solver_type"], u_prec)
        # u_sol.parameters.update(dmn.krylov_solvers)
        self.velocity_krylov_solver = {
            "solver_type": "bicgstab",
            "preconditioner_type": "jacobi",
        }
        self.pressure_krylov_solver = {
            "solver_type": "gmres",
            "preconditioner_type": "hypre_amg",
        }
        self.scalar_krylov_solver = {
            "solver_type": "bicgstab",
            "preconditioner_type": "jacobi",
        }
        self.nut_krylov_solver = {
            "method": "WeightedAverage",  # Or 'default'
            "solver_type": "cg",
            "preconditioner_type": "jacobi",
        }
        self.nu_nn_krylov_solver = {
            "method": "WeightedAverage",  # Or 'default'
            "solver_type": "cg",
            "preconditioner_type": "jacobi",
        }

        self.constrained_domain = None
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
        cd = self.constrained_domain
        mesh = self.mesh
        sys_comp = self.sys_comp
        deg_v, deg_p = self.velocity_degree, self.pressure_degree
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
        self.b_tmp = dict((ui, df.Vector(self.q_[ui].vector())) for ui in sys_comp)
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

    #     # dx, dy = V.tabulate_dof_coordinates().T  # 10942,2
    #     # # dofmap = V.dofmap()  # len(dofmap.dofs()) = 1042
    #     # msh = V.mesh()
    #     # mx, my = msh.coordinates().T  # 2805, 2
    #     # cells = msh.cells()
    #     # fig, ax = plt.subplots()
    #     # # ax.plot(mx, my, "ro")
    #     # plt.triplot(mx, my, cells, "o")
    #     # ax.plot(dx, dy, "k.")
    #     # ax.set_aspect("equal")

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
        if num_iter > 1 and self.print_velocity_pressure_convergence:
            if inner_iter == 1:
                info_blue("  Inner iterations velocity pressure:")
                info_blue("                 error u  error p")
            info_blue(
                "    Iter = {0:4d}, {1:2.2e} {2:2.2e}".format(
                    inner_iter, udiff, df.norm(self.dp_.vector())
                )
            )
