__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import inner, dx, grad, dot, nabla_grad, assemble, norm, normalize
from dolfin import Matrix, Vector, Function, VectorSpaceBasis, Timer
from dolfin import PETScPreconditioner, PETScKrylovSolver, LUSolver
from dolfin import as_backend_type, as_vector
import oasisx.utilities as ut
import matplotlib.pyplot as plt


def attach_pressure_nullspace(Ap, p, Q):
    """Create null space basis object and attach to Krylov solver."""
    null_vec = Vector(p)
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0 / null_vec.norm("l2")
    Aa = as_backend_type(Ap)
    null_space = VectorSpaceBasis([null_vec])
    Aa.set_nullspace(null_space)
    Aa.null_space = null_space


class FirstInner:
    """
    M: mass matrix
    K: stiffness matrix (without viscosity coefficient)
    A: coefficient matrix (needs reassembling)
    """

    def __init__(self, domain):
        u, v = domain.u, domain.v
        dmn = self.domain = domain
        # - - - - - - - - - - - - -SETUP- - - - - - - - - - - - - - - - - -
        # Mass matrix
        M = ut.assemble_matrix(inner(u, v) * dx)
        # Stiffness matrix (without viscosity coefficient)
        K = ut.assemble_matrix(inner(grad(u), grad(v)) * dx)
        if isinstance(dmn.nu, Function):
            dmn.K_scaled = K.copy()
        # Allocate stiffness matrix for LES that changes with time
        KT = (
            None
            if dmn.les_model == "NoModel" and dmn.nn_model == "NoModel"
            else (Matrix(M), inner(grad(u), grad(v)))
        )
        # Pressure Laplacian.
        # Allocate coefficient matrix (needs reassembling)
        A = Matrix(M)
        # Setup for solving convection
        dim = len(dmn.u_components)
        u_ab = as_vector([Function(dmn.VV["u0"]) for i in range(dim)])
        a_conv = inner(v, dot(u_ab, nabla_grad(u))) * dx
        a_scalar = a_conv
        LT = (
            None
            if dmn.les_model == "NoModel"
            else ut.LESsource(dmn.nut_, u_ab, dmn.VV["u0"], name="LTd")
        )

        NT = (
            None
            if dmn.nn_model == "NoModel"
            else ut.NNsource(dmn.nunn_, u_ab, dmn.VV["u0"], name="NTd")
        )
        # for the scalar solver and the first iter
        dmn.K = K
        dmn.M = M
        # for first iter only:
        self.A = A
        self.a_conv = a_conv
        self.a_scalar = a_scalar
        self.LT = LT
        self.KT = KT
        self.NT = NT
        self.u_ab = u_ab
        return

    def assemble(self):
        """Called on first inner iteration of velocity/pressure system.

        Assemble convection matrix, compute rhs of tentative velocity and
        reset coefficient matrix for solve.
        """
        dmn = self.domain
        K = dmn.K
        M = dmn.M
        A = self.A
        a_conv = self.a_conv
        LT = self.LT
        KT = self.KT
        NT = self.NT
        u_ab = self.u_ab

        t0 = Timer("Assemble first inner iter")

        # Update u_ab used as convecting velocity
        for i, ui in enumerate(dmn.u_components):
            u_ab[i].vector().zero()
            u_ab[i].vector().axpy(1.5, dmn.q_1[ui].vector())
            u_ab[i].vector().axpy(-0.5, dmn.q_2[ui].vector())
        # does not need to be assembled. matrix multipl. is enough
        # a_conv from init: inner(v, dot(u_ab, nabla_grad(u))) * dx
        A = assemble(a_conv, tensor=A)
        A *= -0.5  # Negative convection on the rhs
        A.axpy(1.0 / dmn.dt, M, True)  # Add mass
        # Set up scalar matrix for rhs using the same convection as velocity
        if len(dmn.scalar_components) > 0:
            Ta = dmn.Ta  # = 1/dt * M + u_ab
            Ta.zero()
            Ta.axpy(1., A, True)

        # Add diffusion and compute rhs for all velocity components
        if isinstance(dmn.nu, Function):
            K_scaled = dmn.K_scaled
            K_scaled.zero()
            K_scaled.axpy(1., K, True)
            K_scaled.instance().mat().diagonalScale(dmn.nu.vector().vec())
            A.axpy(-0.5, K_scaled, True)
        else:
            A.axpy(-0.5*dmn.nu, K, True)
        if dmn.les_model != "NoModel":
            assemble(dmn.nut_ * KT[1] * dx, tensor=KT[0])
            A.axpy(-0.5, KT[0], True)

        if dmn.nn_model != "NoModel":
            assemble(dmn.nunn_ * KT[1] * dx, tensor=KT[0])
            A.axpy(-0.5, KT[0], True)

        for i, ui in enumerate(dmn.u_components):
            # Start with body force b0
            # TODO: dmn.b_tmp[ui].assign(dmn.b0[ui])
            dmn.b_tmp[ui].zero()
            dmn.b_tmp[ui].axpy(1.0, dmn.b0[ui])
            # Add transient, convection and diffusion
            dmn.b_tmp[ui].axpy(1.0, A * dmn.q_1[ui].vector())
            if dmn.les_model != "NoModel":
                LT.assemble_rhs(i)
                dmn.b_tmp[ui].axpy(1.0, LT.vector())
            if dmn.nn_model != "NoModel":
                NT.assemble_rhs(i)
                dmn.b_tmp[ui].axpy(1.0, NT.vector())

        # Reset matrix for lhs
        A *= -1.0
        A.axpy(2.0 / dmn.dt, M, True)
        [bc.apply(A) for bc in dmn.bcs["u0"]]  # TODO: is this correct?
        t0.stop()
        return


class TentativeVelocityStep:
    def __init__(self, domain):
        dmn = self.domain = domain
        # - - - - - - - - - - - - -SETUP- - - - - - - - - - - - - - - - - -
        # Allocate a dictionary of Functions for holding and computing
        # pressure gradients
        gradp = {}
        p_ = dmn.q_["p"]
        method = dmn.velocity_update_solver
        for i, ui in enumerate(dmn.u_components):
            name = "dpd" + ("x", "y", "z")[i]
            bcs = ut.homogenize(dmn.bcs[ui])
            gradp[ui] = ut.GradFunction(p_, dmn.VV["u0"], i, bcs, name, method)
            # gradp_DG0[ui] = ut.GradFunction(p_, dmn.R, i)  # , bcs, name, method)

        # - - - - - - - - - -get_solvers - - - - - - - - - - - - - - - - - -
        if domain.use_krylov_solvers:
            vks = dmn.velocity_krylov_solver
            u_prec = PETScPreconditioner(vks["preconditioner_type"])
            u_sol = PETScKrylovSolver(vks["solver_type"], u_prec)
            u_sol.parameters.update(dmn.krylov_solvers)
        else:
            u_sol = LUSolver()
        self.u_sol = u_sol
        self.gradp = gradp
        # self.gradp_DG0 = gradp_DG0
        return

    def assemble(self, ui):
        dmn = self.domain
        dmn.b[ui].zero()
        dmn.b[ui].axpy(1.0, dmn.b_tmp[ui])  # b_tmp holds body forces
        self.gradp[ui].assemble_rhs(dmn.q_["p"])
        dmn.b[ui].axpy(-1.0, self.gradp[ui].rhs)
        return

    def solve(self, ui, udiff):
        """Linear algebra solve of tentative velocity component."""
        dmn = self.domain
        [bc.apply(dmn.b[ui]) for bc in dmn.bcs[ui]]
        # q_2 only used on inner_iter 1, so use here as work vector
        dmn.q_2[ui].assign(dmn.q_[ui])
        t1 = Timer("Tentative Linear Algebra Solve")
        self.u_sol.solve(self.A, dmn.q_[ui].vector(), dmn.b[ui])
        t1.stop()
        # udiff += norm(dmn.q_2[ui].vector() - dmn.q_[ui].vector())
        old = dmn.q_2[ui].vector().vec().array
        new = dmn.q_[ui].vector().vec().array
        udiff += ((old - new) ** 2).sum() ** 0.5
        return udiff

    def velocity_update(self, ui):
        """Update the velocity after regular pressure velocity iterations."""
        dmn = self.domain
        # for ui in u_components:
        grad_dp = self.gradp[ui](dmn.dp_)  # grad(p_new - p*)
        # print(grad_dp is self.gradp[ui].vector())
        # u = u* - dt*grad(dp_x); v = v* - dt*grad(dp_y)
        dmn.q_[ui].vector().axpy(-dmn.dt, grad_dp)
        [bc.apply(dmn.q_[ui].vector()) for bc in dmn.bcs[ui]]
        return


class PressureStep:
    def __init__(self, domain):
        q, p = domain.q, domain.p
        dmn = self.domain = domain
        # - - - - - - - - - - - - -SETUP- - - - - - - - - - - - - - - - - -
        # Allocate Function for holding and computing the
        # velocity divergence on Q
        method = dmn.velocity_update_solver
        divu = ut.DivFunction(dmn.u_, dmn.VV["p"], name="divu", method=method)
        # Pressure Laplacian.
        Ap = ut.assemble_matrix(inner(grad(q), grad(p)) * dx, dmn.bcs["p"])
        if dmn.bcs["p"] == []:
            attach_pressure_nullspace(Ap, dmn.q_["p"].vector(), dmn.VV["p"])
        if dmn.use_krylov_solvers:
            # pressure solver ##
            pks = dmn.pressure_krylov_solver
            p_prec = PETScPreconditioner(pks["preconditioner_type"])
            p_sol = PETScKrylovSolver(pks["solver_type"], p_prec)
            p_sol.parameters.update(dmn.krylov_solvers)
            p_sol.set_reuse_preconditioner(True)
        else:
            # pressure solver ##
            p_sol = LUSolver()
        self.divu = divu
        self.Ap = Ap
        self.p_sol = p_sol

    #     gradp_DG0 = {}
    #     for i, ui in enumerate(dmn.u_components):
    #         gradp_DG0[ui] = ut.GradFunction(
    #             dmn.q_["p"], dmn.R, i
    #         )  # , bcs, name, method)
    #     self.gradp_DG0 = gradp_DG0
    #     return

    # def pressure_gradient(self):
    #     dmn = self.domain
    #     res = [None, None]
    #     for i, ui in enumerate(dmn.u_components):
    #         self.gradp_DG0[ui].assemble_rhs(dmn.q_["p"])
    #         res[i] = self.gradp_DG0[ui](dmn.q_["p"])
    #     return res

    def assemble(self):
        """Assemble rhs of pressure equation.
        rhs = -1/dt*inner(div(u), v) *dx + inner(grad(p*), (grad(q)) *dx"""
        dmn = self.domain
        self.divu.assemble_rhs()  # Computes div(u_)*q*dx
        dmn.b["p"][:] = self.divu.rhs
        dmn.b["p"] *= -1.0 / dmn.dt
        # print(dmn.mesh.num_vertices())  # s
        # print(self.Ap.instance().mat().getSize())
        # print(dmn.q_["u0"].vector().vec().getSize())
        dmn.b["p"].axpy(1.0, self.Ap * dmn.q_["p"].vector())
        # print("divu", (self.divu.rhs.get_local() ** 2).sum() ** 0.5)
        return

    def solve(self):
        """Solve pressure equation."""
        dmn = self.domain
        dpv = dmn.dp_.vector()
        p_ = dmn.q_["p"].vector()  # =p*

        [bc.apply(dmn.b["p"]) for bc in dmn.bcs["p"]]
        dpv.zero()
        dpv.axpy(1.0, p_)  # dp_ = 0 + 1.0*p*
        # KrylovSolvers use nullspace for normalization of pressure
        if hasattr(self.Ap, "null_space"):
            self.p_sol.null_space.orthogonalize(dmn.b["p"])

        t1 = Timer("Pressure Linear Algebra Solve")
        # if hasattr(p_approx, "__len__"):
        #     p_.array = p_approx.ravel()
        # else:
        self.p_sol.solve(self.Ap, p_, dmn.b["p"])
        t1.stop()
        # LUSolver use normalize directly for normalization of pressure
        if dmn.bcs["p"] == []:
            normalize(p_)
        dpv.axpy(-1.0, p_)  # dp_ = p* - p_new
        dpv *= -1.0  # dp_ = p_new - p*
        # pdiff = norm(dpv)
        # dpv = dp_ is only used for the velocity correction
        pdiff = ((dpv.vec().array) ** 2).sum() ** 0.5
        return pdiff


class ScalarSolver:
    def __init__(self, domain):
        dmn = self.domain = domain
        M = dmn.M  # from FirstInnerIter
        # ... get_solvers:
        if dmn.use_krylov_solvers:
            # scalar solver ##
            c_prec = PETScPreconditioner(
                dmn.scalar_krylov_solver["preconditioner_type"]
            )
            c_sol = PETScKrylovSolver(
                dmn.scalar_krylov_solver["solver_type"], c_prec
            )
            c_sol.parameters.update(dmn.krylov_solvers)
        else:
            c_sol = LUSolver()

        # ... setup:
        # Allocate coefficient matrix and work vectors for scalars.
        # Matrix differs from velocity in boundary conditions only
        dmn.Ta = Matrix(M)
        if len(dmn.scalar_components) > 1:
            # For more than one scalar we use the same linear algebra
            # solver for all.
            # For this to work we need some additional tensors.
            # The extra matrix is required since different scalars may have
            # different boundary conditions
            Tb = Matrix(M)
            sc0 = dmn.scalar_components[0]
            bb = Vector(dmn.q_[sc0].vector())
            bx = Vector(dmn.q_[sc0].vector())
            self.Tb = Tb
            self.bb = bb
            self.bx = bx

    def assemble(self):
        """Assemble scalar equation."""
        dmn = self.domain
        M = dmn.M  # mass matrix from FirstInnerIter
        K = dmn.K  # stiffness matrix from FirstInnerIter
        Ta = dmn.Ta  # intermediate A from FirstInnerIter
        KT = self.KT  # time dep. stiffness mat for LES from FirstInnerIter

        # # # # - - - scalar_assemble - - - # # #
        # # Just in case you want to use a different scalar convection. We will implement that when its actually needed
        # if self.a_scalar is not self.a_conv:
        #     assemble(self.a_scalar, tensor=Ta)
        #     Ta *= -0.5  # Negative convection on the rhs
        #     Ta.axpy(1.0 / dmn.dt, M, True)  # Add mass

        # Compute rhs for all scalars
        for ci in dmn.scalar_components:
            # Ta = M/dt + .5 a_conv 
            # Add diffusion
            Ta.axpy(-0.5 * dmn.D[ci], K, True)  # Ta =  M/dt - .5 a_conv -0.5*D*K
            if dmn.les_model != "NoModel":
                Ta.axpy(-0.5 / dmn.Schmidt_T[ci], KT[0], True)
            if dmn.nn_model != "NoModel":
                Ta.axpy(-0.5 / dmn.Schmidt_T[ci], KT[0], True)

            # Compute rhs
            dmn.b[ci].zero()
            dmn.b[ci].axpy(1.0, Ta * dmn.q_1[ci].vector())
            dmn.b[ci].axpy(1.0, dmn.b0[ci])  # body forces

            # Subtract diffusion
            Ta.axpy(0.5 * dmn.D[ci], K, True)  # Ta = M/dt - .5 a_conv
            if dmn.les_model != "NoModel":
                Ta.axpy(0.5 / dmn.Schmidt_T[ci], KT[0], True)
            if dmn.nn_model != "NoModel":
                Ta.axpy(0.5 / dmn.Schmidt_T[ci], KT[0], True)

        # Reset matrix for lhs - Note scalar matrix does not contain diffusion, 
        # because it differs for each component
        Ta *= -1.0  # Ta = -M/dt + .5 a_conv
        Ta.axpy(2.0 / dmn.dt, M, True) # Ta = M/dt + .5 a_conv
        return

    def solve(self, ci):
        """Solve scalar equation."""
        dmn = self.domain
        K = dmn.K  # stiffness matrix from FirstInnerIter
        Ta.axpy(0.5 * dmn.D[ci], K, True)  # Ta = M/dt + .5 a_conv + 0.5*D*K
        Ta = dmn.Ta
        if len(dmn.scalar_components[ci]) > 1:
            # Reuse solver for all scalars.
            # This requires the same matrix and vectors to be used by c_sol.
            Tb = self.Tb
            bb = self.bb
            bx = self.bx
            # TODO: use assign()
            Tb.zero()
            Tb.axpy(1.0, Ta, True)
            bb.zero()
            bb.axpy(1.0, dmn.b[ci])
            bx.zero()
            bx.axpy(1.0, dmn.q_[ci].vector())
            [bc.apply(Tb, bb) for bc in dmn.bcs[ci]]
            self.c_sol.solve(Tb, bx, bb)
            dmn.q_[ci].vector().zero()
            dmn.q_[ci].vector().axpy(1.0, bx)

        else:
            [bc.apply(Ta, dmn.b[ci]) for bc in dmn.bcs[ci]]
            self.c_sol.solve(Ta, dmn.q_[ci].vector(), dmn.b[ci])

        if len(dmn.scalar_components) > 1:
            Ta.axpy(-0.5 * dmn.D[ci], K, True)  # Subtract diffusion
        # x_[ci][x_[ci] < 0] = 0.               # Bounded solution
        # x_[ci].set_local(maximum(0., x_[ci].array()))
        # x_[ci].apply("insert")
        return
