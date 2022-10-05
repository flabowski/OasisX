#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:01:22 2022

@author: florianma
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from os import listdir, remove, mkdir, fsdecode  # , rename
from os.path import isfile, join, dirname, exists, expanduser
from datetime import datetime
from shutil import copy2
from oasisx.segregated_domain import SegregatedDomain
from dolfin import Expression, DirichletBC, Mesh, XDMFFile, MeshValueCollection
from dolfin import cpp, grad, ds, inner, dx, div, dot, solve, lhs, rhs, project
from dolfin import Constant, Function, VectorElement, FiniteElement, plot
from dolfin import FunctionSpace, TestFunction, TrialFunction, split, assemble
#from ROM.snapshot_manager import Data
#from low_rank_model_construction.proper_orthogonal_decomposition import row_svd
import json
from oasisx.io import parse_command_line
from fractional_step import FractionalStep


class FreezingCavity(SegregatedDomain):
    """Benchmark Computations of Laminar Flow Around a Cylinder"""

    # http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html
    # run for Um in [0.2, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0]
    # or  for Re in [20., 50., 60., 75.0, 100, 150, 200]
    def __init__(self, config):
        """
        Create the required function spaces, functions and boundary conditions
        for a channel flow problem
        """
        super().__init__()
        # problem parameters
        # case = parameters["case"] if "case" in parameters else 1
        # Umax = cases[case]["Um"]  # 0.3 or 1.5 or 1.5 * np.sin(np.pi * t / 8)
        k = 205  # W/(m K)
        cp = 0.91 * 1000  # kJ/(kg K) *1000 = J/(kg K)
        rho = 2350  # kg /m3
        k_r = 0.00001
        alpha = k / (cp * rho)
        self.bc_dict = {"fluid": 0, "bottom": 1, "right": 2, "top": 3, "left": 4}
        self.t_init = Constant(651)
        self.t_amb = Constant(640)
        # self.t_feeder = Constant(670)
        self.k_top = Constant(0.0)
        self.k_lft = Constant(0.0)
        self.k_btm = Constant(0.0)
        self.k_rgt = Constant(k_r*0)
        self.g = Constant((0.0, -9.81))  # used in body_force()
        self.dt = 1.0
        self.T = 10000
        self.scalar_components = ["t"]
        self.D = {"t": Constant(alpha)}

        self.pkg_dir = dirname(__file__).split("oasis")[0]
        self.simulation_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_dir = temp_dir = expanduser("~") + "/tmp/"
        msg = "is not empty. Do you want to remove all its content? [y/n]:"
        if exists(temp_dir):
            if len(listdir(temp_dir)) > 0:
                if input(temp_dir + msg) == "y":
                    for file in listdir(temp_dir):
                        filename = fsdecode(file)
                        remove(temp_dir + filename)
                else:
                    raise ValueError(temp_dir + "needs to be empty")
        else:
            mkdir(temp_dir)

        self.set_parameters(config)
        if "frequency" in config.keys():
            self.dt = 1.0 / config["frequency"]

        m = int(self.T / self.dt)
        self.udiff = np.zeros((m, self.max_iter))
        self.pdiff = np.zeros((m, self.max_iter))
        self.t_u = np.empty((m,))
        self.t_p = np.empty((m,))
        return

    def stokes(self):
        P2 = VectorElement("CG", self.mesh.ufl_cell(), 2)
        P1 = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        TH = P2 * P1
        VQ = FunctionSpace(self.mesh, TH)
        mf = self.mf
        no_slip = Constant((0.0, 0))
        topflow = Constant((0.0, 0))
        bc0 = DirichletBC(VQ.sub(0), topflow, mf, self.bc_dict["top"])
        bc1 = DirichletBC(VQ.sub(0), no_slip, mf, self.bc_dict["left"])
        bc2 = DirichletBC(VQ.sub(0), no_slip, mf, self.bc_dict["bottom"])
        bc3 = DirichletBC(VQ.sub(0), no_slip, mf, self.bc_dict["right"])
        bc4 = DirichletBC(VQ.sub(1), Constant(0), mf, self.bc_dict["top"])
        bcs = [bc0, bc1, bc2, bc3, bc4]

        vup = TestFunction(VQ)
        up = TrialFunction(VQ)
        up_ = Function(VQ)  # Function holding the solution

        u, p = split(up)  # Trial
        vu, vp = split(vup)  # Test
        rho_0 = self.rho.vector().vec().array.mean()
        F = (
            - self.mu/rho_0 * inner(grad(vu), grad(u)) * dx
            + inner(div(vu), p) * dx  # solve for p/rho_0!
            + inner(vp, div(u)) * dx
            + dot(self.g*self.rho, vu)/rho_0 * dx
        )
        solve(lhs(F) == rhs(F), up_, bcs=bcs)
        self.q_["u0"].vector().vec().array[:] = project(up_.sub(0).sub(0), self.VV["u0"]).vector().vec().array[:]
        self.q_["u1"].vector().vec().array[:] = project(up_.sub(0).sub(1), self.VV["u1"]).vector().vec().array[:]
        self.q_["p"].vector().vec().array[:] = project(up_.sub(1), self.VV["p"]).vector().vec().array[:]

        print("mu_max:", self.mu.vector().vec().array[:].max())
        print("u0_max:", self.q_["u0"].vector().vec().array[:].max())
        print("u1_max:", self.q_["u1"].vector().vec().array[:].max())
        
        self.plot()
        plt.savefig(self.temp_dir + "_init.png", dpi=300)
        plt.close()
        return

    def declare_coefficients(self):
        V, Q = self.VV["t"], self.VV["p"]
        self.mu, self.nu, self.rho = Function(V), Function(V), Function(V)
        return

    def assemble_body_force(self):
        print("updating body forces")
        rho_0 = self.rho.vector().vec().array.mean()
        self.f = [self.rho/rho_0*self.g[0], self.rho/rho_0*self.g[1]]
        super().assemble_body_force()
        return

    def initialize_components(self):
        self.t_ = self.q_["t"]
        self.t_1 = self.q_1["t"]
        self.t_.vector().vec().array[:] = self.t_init
        self.t_1.vector().vec().array[:] = self.t_init

        g = self.g
        for ui in self.u_components:
            self.q_[ui].vector().vec().array[:] = 1e-6
            self.q_1[ui].vector().vec().array[:] = 1e-6
            self.q_2[ui].vector().vec().array[:] = 1e-6


        x, y = self.VV["u0"].tabulate_dof_coordinates().T
        self.q_["t"].vector().vec().array = self.t_init.values() - x * .1
        self.q_1["t"].vector().vec().array = self.t_init.values() - x * .1
        self.update_coefficients()
        # initial pressure = static pressure
        xyz = self.VV["p"].tabulate_dof_coordinates().T
        rho_0 = self.rho.vector().vec().array.mean()
        g_z = np.sum(xyz * g.values()[:, None], axis=0)
        # rho is incorporated in the pressure
        self.q_["p"].vector().vec().array[:] = -g_z/rho_0
        self.q_1["p"].vector().vec().array[:] = -g_z/rho_0
        self.stokes()
        self.advance()
        self.stokes()
        self.advance()
        return

    def create_bcs(self):
        mf, bc_dict = self.mf, self.bc_dict
        V, Q = self.VV["u0"], self.VV["p"]
        no_slip = Constant(0.0)
        bc0 = DirichletBC(V, no_slip, mf, bc_dict["top"])
        bc1 = DirichletBC(V, no_slip, mf, bc_dict["left"])
        bc2 = DirichletBC(V, no_slip, mf, bc_dict["bottom"])
        bc3 = DirichletBC(V, no_slip, mf, bc_dict["right"])
        bcu = [bc0, bc1, bc2, bc3]
        bcp = [DirichletBC(Q, Constant(0), mf, bc_dict["top"])]
        # bcp = []

        lft, rgt = bc_dict["left"], bc_dict["right"]
        top, btm = bc_dict["top"], bc_dict["bottom"]
        v, ds_, t = self.v, self.ds_, self.u  # we use the same trial function as 
        t1, t_amb = self.q_1["t"], self.t_amb
        # ... = ... - k (t-t_amb) * v * ds; with t = (t^n+t^{n-1})/2
        # t^n * (k*v*ds) = - t^{n-1} * (k*v*ds) + t_amb * (k*v*ds)
        self.bt_lhs = 1.0 * assemble(  # 0.5 for c-n
            + self.k_lft * t * v * ds_(lft)
            + self.k_rgt * t * v * ds_(rgt)
            + self.k_top * t * v * ds_(top)
            + self.k_btm * t * v * ds_(btm)
        )
        self.bt_rhs = assemble(
            + self.k_lft * (-0.0 * t1+t_amb) * v * ds_(lft)  # 0.5 for c-n
            + self.k_rgt * (-0.0 * t1+t_amb) * v * ds_(rgt)
            + self.k_top * (-0.0 * t1+t_amb) * v * ds_(top)
            + self.k_btm * (-0.0 * t1+t_amb) * v * ds_(btm)
        )
        # indexptr, indices, data = self.bt_lhs.instance().mat().getValuesCSR()
        # print(indexptr.shape, indices.shape, data.shape)
        # print(self.bt_rhs.instance().vec().array.shape)

        self.bcs = {
            "u0": bcu,
            "u1": bcu,
            "p": bcp,
            "t": [],
        }
        return

    def mu_(self, T):
        mu_liquidus = 1.3 / 1000.  # in Pa*s; Water at 20°C: 1 mPa s = 1/1000 Pa s
        mu = np.array([(T - 650) ** 2 * 1000000.0]).ravel() + mu_liquidus
        mu[T > 650] = mu_liquidus
        return mu

    def rho_(self, T):
        """
        # see: https://www.epj-conferences.org/articles/epjconf/pdf/2011/05/epjconf_lam14_01024.pdf
        Table 4 in Viscosity and volume properties of the Al-Cu melts.
        N. Konstantinova, A. Kurochkin, and P. Popel
        """
        t = np.array([0.00, 700., 750., 800., 850., 900.,
                                950., 1000, 1050, 1100, 1150, 1200,
                                1250, 1300, 1350, 1400, 1450, 1500])
        r = np.array([2380.0, 2351.5, 2340.6, 2329.8, 2318.9, 2308.1,
                            2297.2, 2286.3, 2275.5, 2264.6, 2253.8, 2242.9,
                            2232.1, 2221.2, 2210.4, 2199.5, 2188.6, 2177.8])
        # t = np.array([0.00, 660, 670, 1000])
        # r = np.array([998.8, 998.8, 998.7, 998.7])
        f_rho = interp1d(
            t, r, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        return f_rho(T)  # kg/m3

    def update_coefficients(self):
        temperature_field = self.get_t()
        mu_updated = self.mu_(temperature_field)
        self.set_mu(mu_updated)
        rho_updated = self.rho_(temperature_field)
        self.set_rho(rho_updated)
        rho_0 = self.rho.vector().vec().array.mean()
        nu_updated = mu_updated / rho_0
        self.set_nu(nu_updated)
        self.assemble_body_force()
        return

    def advance(self):
        # nu is temperature dependant.
        # But we want to solve the scalar equation only once for t
        # Therfore, nu is not listed in scalar_components
        # and thus needs to be handled explicitly here.
        super().advance()
        self.update_coefficients()
        return

    def get_rho(self):
        return self.rho.vector().vec().array

    def set_rho(self, rho):
        self.rho.vector().vec().array[:] = rho

    def get_mu(self):
        return self.mu.vector().vec().array

    def set_mu(self, mu):
        self.mu.vector().vec().array[:] = mu

    def get_nu(self):
        return self.nu.vector().vec().array

    def set_nu(self, nu):
        self.nu.vector().vec().array[:] = nu

    def get_t(self):
        return self.t_.vector().vec().array

    def set_t(self, t):
        self.t_.vector().vec().array[:] = t

    # def update_bcs(self, t):
    #     """
    #     ...
    #     note: expressions are cached on the hard drive. call $dijitso show
    #     dijitso config

    #     """
    #     self.update_coefficients()
    #     return

    def start_timestep_hook(self, t, **kvargs):
        """Called at start of new timestep"""
        # self.update_bcs(t)
        pass

    def temporal_hook(self, t, tstep, ps, **kvargs):
        i = tstep - 1
        pth = self.temp_dir
        mesh = self.mesh
        u0 = self.q_["u0"].compute_vertex_values(mesh)
        u1 = self.q_["u1"].compute_vertex_values(mesh)
        nu = self.nu.compute_vertex_values(mesh)
        u = (u0 ** 2 + u1 ** 2) ** 0.5
        L = 1
        Re = u*L/nu
        C = u * self.dt/mesh.hmax()
        print("Re, CFL", Re.max(), C.max())

        if (i % self.plot_interval) == 0 or (t + 1e-6) > self.T or (i<3):
            fig, axs = self.plot()
            plt.suptitle("t = {:.2f} s".format(t))
            plt.savefig(pth + "frame_{:06d}.png".format(i), dpi=300)
            plt.close()


        if (i % self.save_step) == 0:
            # u = self.q_["u0"].compute_vertex_values(mesh)  # 2805
            u = self.q_["u0"].vector().vec().array  # 10942
            v = self.q_["u1"].vector().vec().array
            p = self.q_["p"].vector().vec().array
            t = self.q_["t"].vector().vec().array
            #dpx, dpy = ps.pressure_gradient()
            np.save(pth + "u_{:06d}.npy".format(i), u)
            np.save(pth + "v_{:06d}.npy".format(i), v)
            np.save(pth + "p_{:06d}.npy".format(i), p)
            np.save(pth + "t_{:06d}.npy".format(i), t)
            # np.save(pth + "dpx_{:06d}.npy".format(i), dpx)
            # np.save(pth + "dpy_{:06d}.npy".format(i), dpy)
            # tvs = kvargs.get("tvs", None)
            # if tvs:
            #     # TODO
            #     gradpx = tvs.gradp["u0"].vector().vec().array
            #     gradpy = tvs.gradp["u1"].vector().vec().array
            #     np.save(pth + "gradpx_{:06d}.npy".format(i), gradpx)
            #     np.save(pth + "gradpy_{:06d}.npy".format(i), gradpy)
        return

    def scalar_hook(self):
        # print("skipping scalar_hook")
        return

    def theend_hook(self, SVD=False):
        print("post processing:")
        pth = self.pkg_dir + "results/" + self.simulation_start + "/"
        mkdir(pth)
        tmp_pth = self.temp_dir
        # save meshes as well as some other data
        V, Q = self.VV["u0"], self.VV["p"]
        # np.save(pth + "drag.npy", self.C_D)
        # np.save(pth + "lift.npy", self.C_L)
        np.save(pth + "V_dof_coords.npy", V.tabulate_dof_coordinates())
        np.save(pth + "Q_dof_coords.npy", Q.tabulate_dof_coordinates())
        np.save(pth + "mesh_coords.npy", V.mesh().coordinates())
        np.save(pth + "mesh_cells.npy", V.mesh().cells())
        t = np.arange(0.0, self.T, self.dt) + self.dt
        np.save(pth + "time.npy", t)
        # np.save(pth + "nu.npy", np.array([self.nu]))
        # np.save(pth + "Re.py", np.array([self.Re]))
        mesh_dir = dirname(self.mesh_name) + "/"
        mesh_name = self.mesh_name.split("/")[-1].replace(".xdmf", "")
        copy2(mesh_dir + mesh_name + ".xdmf", pth + mesh_name + ".xdmf")
        copy2(mesh_dir + mesh_name + ".h5", pth + mesh_name + ".h5")
        facet_dir = dirname(self.facet_name) + "/"
        facet_name = self.facet_name.split("/")[-1].replace(".xdmf", "")
        copy2(facet_dir + facet_name + ".xdmf", pth + facet_name + ".xdmf")
        copy2(facet_dir + facet_name + ".h5", pth + facet_name + ".h5")

        if hasattr(self, "udiff"):
            np.save(pth + "udiff.npy", self.udiff)
        if hasattr(self, "pdiff"):
            np.save(pth + "pdiff.npy", self.pdiff)
        # for nm in ["mf.xdmf", "mf.h5", "mesh.xdmf", "mesh.h5"]:
        #     src = self.pkg_dir + nm
        #     dst = pth + nm
        #     print(src, dst)
        #     copy2(src, dst)
        # move temp png files to results folder
        for quantity in ["p", "t", "u", "v"]:
            onlyfiles = [
                f
                for f in listdir(tmp_pth)
                if (isfile(join(tmp_pth, f)) and f.startswith(quantity + "_"))
            ]
            onlyfiles.sort()

            u = np.load(tmp_pth + onlyfiles[0])
            X_q = np.empty((len(u), len(onlyfiles)))
            for i, f in enumerate(onlyfiles):
                X_q[:, i] = np.load(tmp_pth + f)
            print(quantity, X_q.min(), X_q.max(), X_q.mean())
            np.save(pth + "X_" + quantity + ".npy", X_q)

            #my_data = Data(X_q, False)
            #X_n = my_data.normalise()
            # if SVD:
            #     print("SVD of a ", X_n.shape, "matrix:")
            #     c = max(1, X_n.shape[1] // 2000)
            #     # U, S, VT = row_svd(X_n, 1, 1, False, False)
            #     U, S, VT = row_svd(X_n, c, 1 - 1e-6, True, True)
            #     np.save(pth + "ROM_U_" + quantity + ".npy", U)
            #     np.save(pth + "ROM_S_" + quantity + ".npy", S)
            #     np.save(pth + "ROM_VT_" + quantity + ".npy", VT)
            #     np.save(pth + "ROM_X_min_" + quantity + ".npy", my_data.X_min)
            #     np.save(pth + "ROM_X_range_" + quantity + ".npy", my_data.X_range)
            for i, f in enumerate(onlyfiles):
                remove(tmp_pth + f)
        # move temp png files to results folder
        for f in listdir(tmp_pth):
            if isfile(join(tmp_pth, f)) and f.endswith(".png"):
                copy2(tmp_pth + f, pth + f)
                remove(tmp_pth + f)
        if SVD:
            self.plot_decay()
        print("found", len(t), "timesteps and", len(onlyfiles), "saved files")
        return

    def plot_decay(self):
        pth = self.pkg_dir + "results/" + self.simulation_start + "/"
        plot_width = 8
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(plot_width / 2.54 * 2, plot_width / 2.54)
        )
        # prefixes = ["p", "u", "v", "gradpx", "gradpy"]
        prefixes = ["p", "dpx", "dpy", "u", "v", "gradpx", "gradpy"]
        for i, quantity in enumerate(prefixes):  # u, v, p, t, r, m]):
            S = np.load(pth + "ROM_S_" + quantity + ".npy")
            col = colors[i]
            lbl = prefixes[i]
            ax1.plot(np.arange(0, len(S)), S, color=col, marker=".")
            ax2.plot(
                np.arange(0, len(S)),
                np.cumsum(S) / S.sum() * 100,
                color=col,
                marker=".",
                label=lbl,
            )
        ax1.set_xlabel("rank r")
        ax1.set_ylabel("singular values")
        ax2.set_xlabel("rank r")
        ax1.set_yscale("log")
        ax1.set_ylim([1e-8, 1e3])
        ax2.set_ylim([95, 100])
        ax2.set_ylabel("Cumulative Energy [%]")
        ax2.legend()
        ax1.grid(which="both")
        # plt.title("dacay of singular values")
        ax1.set_xlim([0, 1000])
        ax2.set_xlim([0, 200])
        plt.grid()
        plt.tight_layout()
        # plt.show()
        plt.savefig(pth + "singular_values.png")
        plt.close()

    def plot(self):
        mesh = self.mesh
        u = self.q_["u0"].compute_vertex_values(mesh)
        v = self.q_["u1"].compute_vertex_values(mesh)
        p_ = self.q_["p"].compute_vertex_values(mesh)
        t = self.q_["t"].compute_vertex_values(mesh)
        rho = self.rho.compute_vertex_values(mesh).copy()
        mu = self.mu.compute_vertex_values(mesh).copy()
        nu = self.nu.compute_vertex_values(mesh).copy()
        p = p_*rho
        # nu[nu>0.8] = 0.8
        # print(u.shape, v.shape, p.shape)
        # self.b0["u0"].instance().vec().array
        # x2, y2 = V.tabulate_dof_coordinates().T
        # u = self.b_tmp["u0"].instance().vec().array
        # v = self.b_tmp["u1"].instance().vec().array
        # print(self.b_tmp["u1"].instance().vec().array.min())
        magnitude = (u ** 2 + v ** 2) ** 0.5
        # print(u.shape, v.shape, p.shape, magnitude.shape)

        # velocity = u.compute_vertex_values(mesh)
        # velocity.shape = (2, -1)
        # magnitude = np.linalg.norm(velocity, axis=0)
        x, y = mesh.coordinates().T
        # u, v = velocity
        tri = mesh.cells()
        # pressure = p.compute_vertex_values(mesh)
        # print(x.shape, y.shape, u.shape, v.shape)
        fs = (12, 6)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=fs)
        c1 = ax1.quiver(x, y, u, v, magnitude)
        c2 = ax2.tricontourf(x, y, tri, p, levels=40)
        c3 = ax3.tricontourf(x, y, tri, t, levels=40)
        c4 = ax4.tricontourf(x, y, tri, rho, levels=40)
        # ax4.plot(t, nu, "b.")
        # print(p.min(), p.max())
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax3.set_aspect("equal")
        ax4.set_aspect("equal")
        ax1.set_title("velocity")
        ax2.set_title("pressure")
        ax3.set_title("temperature")
        ax4.set_title("density")
        plt.colorbar(c1, ax=ax1)
        plt.colorbar(c2, ax=ax2)
        plt.colorbar(c3, ax=ax3)
        plt.colorbar(c4, ax=ax4)
        return fig, (ax1, ax2, ax3)


if __name__ == "__main__":
    pkg_dir = dirname(__file__).split("oasisx")[0]
    path_to_config = pkg_dir + "/resources/freezing_cavity/"
    config_file = path_to_config + "config.json"
    print(config_file)
    config = json.load(open(file=config_file, encoding="utf-8"))
    commandline_args = parse_command_line()

    my_domain = FreezingCavity(config)
    my_domain.set_parameters(commandline_args)
    #my_domain.mesh_name = pkg_dir + "/resources/freezing_cavity/mesh.xdmf"
    my_domain.mesh_from_file()
    
    algorithm = FractionalStep(my_domain)
    my_domain.plot()
    algorithm.run()
    pth = pkg_dir + "results/" + my_domain.simulation_start + "/"
    copy2(config_file, pth + config_file.split("/")[-1])
