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

    def __init__(self, domain, config):
        u, v = domain.u, domain.v
        dmn = self.domain = domain
        self.config = config
        # - - - - - - - - - - - - -SETUP- - - - - - - - - - - - - - - - - -
        # Mass matrix
        M = ut.assemble_matrix(inner(u, v) * dx)
        # Stiffness matrix (without viscosity coefficient)
        K = ut.assemble_matrix(inner(grad(u), grad(v)) * dx)
        if isinstance(dmn.nu, Function):
            dmn.K_scaled = K.copy()
        # Allocate stiffness matrix for LES that changes with time
        # Pressure Laplacian.
        # Allocate coefficient matrix (needs reassembling)
        A = Matrix(M)
        # Setup for solving convection
        dim = len(dmn.u_components)
        u_ab = as_vector([Function(dmn.VV["u0"]) for i in range(dim)])
        a_conv = inner(v, dot(u_ab, nabla_grad(u))) * dx
        a_scalar = a_conv
        # for the scalar solver and the first iter
        dmn.K = K
        dmn.M = M
        # for first iter only:
        self.A = A
        self.a_conv = a_conv
        self.a_scalar = a_scalar
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
        # Add mass, A=-0.5*a_conv+1/dt*M
        A.axpy(1.0 / self.config["dt"], M, True)
        # A.axpy(1.0 / self.config["dt"], M, True)  # Add mass, A=-0.5*a_conv+1/dt*M
        # Set up scalar matrix for rhs using the same convection as velocity
        if len(dmn.scalar_components) > 0:
            Ta = dmn.Ta  # = 1/dt * M -.5* u_ab
            Ta.zero()
            Ta.axpy(1.0, A, True)
        # Add diffusion and compute rhs for all velocity components
        if isinstance(dmn.nu, Function):
            print("scaling by nu")
            K_scaled = dmn.K_scaled
            K_scaled.zero()
            K_scaled.axpy(1.0, K, True)
            K_scaled.instance().mat().diagonalScale(dmn.nu.vector().vec())
            A.axpy(-0.5, K_scaled, True)

            # re-use K_scaled to add velocity penalization
            K_scaled.zero()
            K_scaled.axpy(1.0, M, True)
            Phi_s = 1 - dmn.phi_l.vector().vec()
            K_scaled.instance().mat().diagonalScale(-Phi_s / dmn.config["tau"])
            A.axpy(0.5, K_scaled, True)
        else:
            A.axpy(-0.5 * dmn.nu, K, True)
        for i, ui in enumerate(dmn.u_components):
            # Start with body force b0
            # TODO: dmn.b_tmp[ui].assign(dmn.b0[ui])
            dmn.b_tmp[ui].zero()
            dmn.b_tmp[ui].axpy(1.0, dmn.b0[ui] * dmn.f[ui])
            # Add transient, convection and diffusion
            # =b0 +(1/dt*M-0.5*a_conv-0.5*nu*K )*u
            dmn.b_tmp[ui].axpy(1.0, A * dmn.q_1[ui].vector())
        # Reset matrix for lhs
        A *= -1.0
        A.axpy(2.0 / self.config["dt"], M, True)
        [bc.apply(A) for bc in dmn.bcs["u0"]]  # TODO: is this correct?
        t0.stop()
        return


class TentativeVelocityStep:
    def __init__(self, domain, config):
        # dmn = my_domain
        # config = solver_config
        dmn = self.domain = domain
        self.config = config
        # slv = self.solver = solver
        # - - - - - - - - - - - - -SETUP- - - - - - - - - - - - - - - - - -
        # Allocate a dictionary of Functions for holding and computing
        # pressure gradients
        gradp = {}
        p_ = dmn.q_["p"]
        method = config["velocity_update_solver"]
        for i, ui in enumerate(dmn.u_components):
            name = "dpd" + ("x", "y", "z")[i]
            bcs = ut.homogenize(dmn.bcs[ui])
            gradp[ui] = ut.GradFunction(p_, dmn.VV["u0"], i, bcs, name, method)
        # - - - - - - - - - -get_solvers - - - - - - - - - - - - - - - - - -
        if config["use_krylov_solvers"]:
            p_type = config["velocity_krylov_solver"]["preconditioner_type"]
            s_type = config["velocity_krylov_solver"]["solver_type"]
            u_prec = PETScPreconditioner(p_type)
            u_sol = PETScKrylovSolver(s_type, u_prec)
            u_sol.parameters.update(config["krylov_solvers"])
        else:
            u_sol = LUSolver()
        self.u_sol = u_sol
        self.gradp = gradp
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
        dmn.q_[ui].vector().axpy(-self.config["dt"], grad_dp)
        [bc.apply(dmn.q_[ui].vector()) for bc in dmn.bcs[ui]]
        return


class PressureStep:
    def __init__(self, domain, config):
        q, p = domain.q, domain.p
        dmn = self.domain = domain
        self.config = config
        # - - - - - - - - - - - - -SETUP- - - - - - - - - - - - - - - - - -
        # Allocate Function for holding and computing the
        # velocity divergence on Q
        method = config["velocity_update_solver"]
        divu = ut.DivFunction(dmn.u_, dmn.VV["p"], name="divu", method=method)
        # Pressure Laplacian.
        Ap = ut.assemble_matrix(inner(grad(q), grad(p)) * dx, dmn.bcs["p"])
        if dmn.bcs["p"] == []:
            attach_pressure_nullspace(Ap, dmn.q_["p"].vector(), dmn.VV["p"])
        if config["use_krylov_solvers"]:
            # pressure solver ##
            # pks = solver_config.pressure_krylov_solver
            p_type = config["pressure_krylov_solver"]["preconditioner_type"]
            s_type = config["pressure_krylov_solver"]["solver_type"]
            p_prec = PETScPreconditioner(p_type)
            p_sol = PETScKrylovSolver(s_type, p_prec)
            p_sol.parameters.update(config["krylov_solvers"])
            p_sol.set_reuse_preconditioner(True)
        else:
            # pressure solver ##
            p_sol = LUSolver()
        self.divu = divu
        self.Ap = Ap
        self.p_sol = p_sol
        return

    def assemble(self):
        """Assemble rhs of pressure equation.
        rhs = -1/dt*inner(div(u), v) *dx + inner(grad(p*), (grad(q)) *dx"""
        dmn = self.domain
        self.divu.assemble_rhs()  # Computes div(u_)*q*dx
        dmn.b["p"][:] = self.divu.rhs
        dmn.b["p"] *= -1.0 / self.config["dt"]
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
        pdiff = ((dpv.vec().array) ** 2).sum() ** 0.5
        return pdiff


class ScalarSolver:
    def __init__(self, domain, config):
        dmn = self.domain = domain
        self.config = config
        M = dmn.M  # from FirstInnerIter
        # ... get_solvers:
        if config["use_krylov_solvers"]:
            # scalar solver ##
            p_type = config["scalar_krylov_solver"]["preconditioner_type"]
            s_type = config["scalar_krylov_solver"]["solver_type"]
            c_prec = PETScPreconditioner(p_type)
            c_sol = PETScKrylovSolver(s_type, c_prec)
            c_sol.parameters.update(config["krylov_solvers"])
        else:
            c_sol = LUSolver()
        self.c_sol = c_sol
        # ... setup:
        # Allocate coefficient matrix and work vectors for scalars.
        # Matrix differs from velocity in boundary conditions only
        dmn.Ta = Matrix(M)  # just to declare it. fit.assemble sets it to
        # Ta = 1/dt * M -.5* u_ab
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
        # if k == 0:
        #     dmn.phi_l.assign(dmn.phi_l_1)  #

        # Compute rhs for all scalars
        for ci in dmn.scalar_components:
            # Ta = M/dt + .5 a_conv
            # Add diffusion
            Ta.axpy(-0.5 * dmn.D[ci], K, True)
            # Ta =  M/dt - .5 a_conv  -0.5*D*K
            # Compute rhs: b = t_1 * (M/dt - .5 a_conv -0.5*D*K)
            dmn.b[ci].zero()
            dmn.b[ci].axpy(1.0, Ta * dmn.q_1[ci].vector())
            dmn.b[ci].axpy(1.0, dmn.b0[ci])  # body forces
            # Subtract diffusion
            Ta.axpy(0.5 * dmn.D[ci], K, True)  # Ta = M/dt - .5 a_conv
        # Reset matrix for lhs - Note scalar matrix does not contain diffusion,
        # because it differs for each component
        Ta *= -1.0  # Ta = -M/dt + .5 a_conv
        Ta.axpy(2.0 / self.config["dt"], M, True)  # Ta = M/dt + .5 a_conv
        return

    def solve(self, ci, k):
        """Solve scalar equation."""
        dmn = self.domain
        K = dmn.K  # stiffness matrix from FirstInnerIter
        Ta = dmn.Ta
        Ta.axpy(0.5 * dmn.D[ci], K, True)  # Ta = M/dt + .5 a_conv + 0.5*D*K
        # Ta.axpy(0.5, dmn.kvds, True)  # Ta =  M/dt + .5 a_conv + 0.5*D*K + 0.5 k v ds
        if len(dmn.scalar_components) > 1:
            # Reuse solver for all scalars.
            # This requires the same matrix and vectors to be used by c_sol.
            Tb = self.Tb
            bb = self.bb
            bx = self.bx
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
            ddH = dmn.phi_l_1.vector() - dmn.phi_l.vector()
            ddH = ddH / self.config["dt"]
            S = ddH / self.config["c_p"]
            print(k, S.vec().array.min(), S.vec().array.max())
            # mesh = dmn.mesh
            # phi_l = dmn.phi_l.compute_vertex_values(mesh)

            [bc.apply(Ta, dmn.b[ci]) for bc in dmn.bcs[ci]]
            A = Ta + dmn.bt_lhs
            b = dmn.b[ci] + dmn.bt_rhs  # + S
            res = self.c_sol.solve(A, dmn.q_[ci].vector(), b)
            print("scalar solver finished with ", res)

        if len(dmn.scalar_components) > 1:
            Ta.axpy(-0.5 * dmn.D[ci], K, True)  # Subtract diffusion
        return
