#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:01:22 2022

@author: florianma
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from os import listdir, remove, mkdir, fsdecode  # , rename
from os.path import isfile, join, dirname, exists, expanduser
from datetime import datetime
from shutil import copy2
from oasisx.segregated_domain import SegregatedDomain
import json
from dolfin import Expression, DirichletBC, Mesh, XDMFFile, MeshValueCollection
from dolfin import cpp, FacetNormal, grad, Identity, ds, assemble, dot
from oasisx.io import parse_command_line
from fractional_step import FractionalStep

# import oasis.common.utilities as ut

H = 0.41
L = 2.2
D = 0.1
center = 0.2
cases = {1: {"Um": 0.3, "Re": 20.0}, 2: {"Um": 1.5, "Re": 100.0}}


class Cylinder(SegregatedDomain):
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
        self.H = 0.41
        self.mu = 0.001
        self.rho = 1.0
        # self.Schmidt = {}
        # self.Schmidt_T = {}
        self.T = 8
        self.dt = 1 / 1600
        self.scalar_components = []
        self.bc_dict = {
            "fluid": 0,
            "channel_walls": 1,
            "cylinderwall": 2,
            "inlet": 3,
            "outlet": 4,
        }

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
        if "Umax_str" in config.keys():
            self.Umax = self.get_Umax()
            print(self.Umax)
        self.nu = self.mu / self.rho
        self.Umean = 2.0 / 3.0 * self.Umax
        self.coeff = 2 / (self.rho * D * self.Umean ** 2)
        self.Re = self.rho * self.Umean * D / self.mu
        print("Re", self.Re)
        m = int(self.T / self.dt)
        self.udiff = np.zeros((m, self.max_iter))
        self.pdiff = np.zeros((m, self.max_iter))
        self.C_D = np.empty((m,))
        self.C_L = np.empty((m,))
        self.t_u = np.empty((m,))
        self.t_p = np.empty((m,))
        self.C_D[:] = self.C_L[:] = np.NaN
        return

    def get_Umax(self):
        namespace = {
            "sin": np.sin,
            "pi": np.pi,
            "t": np.arange(0, self.T, self.dt),
        }
        return eval(self.Umax_str, namespace).max()

    def create_bcs(self):
        mf, bc_dict = self.mf, self.bc_dict
        V, Q, Umax_str, H = self.VV["u0"], self.VV["p"], self.Umax_str, self.H
        U0_str = Umax_str + "*4.0*x[1]*({0}-x[1])/{1:.6f}".format(H, H ** 2)
        self.inlet = Expression(U0_str, degree=2, t=0.0)
        bc00 = DirichletBC(V, self.inlet, mf, bc_dict["inlet"])
        bc01 = DirichletBC(V, 0.0, mf, bc_dict["inlet"])
        bc10 = bc11 = DirichletBC(V, 0.0, mf, bc_dict["cylinderwall"])
        bc2 = DirichletBC(V, 0.0, mf, bc_dict["channel_walls"])
        bcp = DirichletBC(Q, 0.0, mf, bc_dict["outlet"])
        self.bcs = {
            "u0": [bc00, bc10, bc2],
            "u1": [bc01, bc11, bc2],
            "p": [bcp],
        }
        return

    def update_bcs(self, t):
        """
        note: expressions are cached on the hard drive. call $dijitso show
        dijitso config

        """
        self.inlet.t = t
        inlet = self.bc_dict["inlet"]
        bc00 = DirichletBC(self.VV["u0"], self.inlet, self.mf, inlet)
        self.bcs["u0"][0] = bc00
        return

    def normal_stresses(self):
        p = self.q_["p"]
        n = FacetNormal(self.mesh)
        grad_u = grad(self.u_)
        tau = self.mu * (grad_u + grad_u.T) - p * Identity(2)
        F_drag = assemble(dot(tau, n)[0] * self.ds_(2))
        F_lift = assemble(dot(tau, n)[1] * self.ds_(2))
        return -self.coeff * F_drag, -self.coeff * F_lift

    def start_timestep_hook(self, t, **kvargs):
        """Called at start of new timestep"""
        self.update_bcs(t)
        pass

    def temporal_hook(self, t, tstep, ps, **kvargs):
        i = tstep - 1
        pth = self.temp_dir
        # ts = OasisTimer("normal stresses")
        self.C_D[i], self.C_L[i] = self.normal_stresses()
        # ts.stop()
        self.t_u[i] = t
        if (i % self.plot_interval) == 0 or (t + 1e-6) > self.T:
            u = self.q_["u0"].vector().vec().array  # 10942
            v = self.q_["u1"].vector().vec().array
            p = self.q_["p"].vector().vec().array

            fig, (ax1, ax2) = self.plot()
            plt.savefig(pth + "frame_{:06d}.png".format(i))
            plt.close()

            pth2 = self.pkg_dir + "/resources/cylinder/"
            turek = np.loadtxt(pth2 + "bdforces_lv4")
            t_u, C_D, C_L = self.t_u, self.C_D, self.C_L
            num_velocity_dofs = len(u) + len(v)
            num_pressure_dofs = len(p)
            dofs = num_velocity_dofs + num_pressure_dofs

            plt.figure(figsize=(25, 8))
            lbl = r"FEniCSx  ({0:d} dofs)".format(dofs)
            plt.plot(t_u, C_D, "-.", label=lbl, linewidth=2)
            x, y, lbl = turek[1:, 1], turek[1:, 3], "FEATFLOW (42016 dofs)"
            plt.plot(x, y, marker="x", markevery=50, ls="", ms=4, label=lbl)
            plt.title("Drag coefficient")
            plt.grid()
            plt.legend()
            plt.savefig(pth + "drag_comparison.png")
            plt.close()

            plt.figure(figsize=(25, 8))
            lbl = r"FEniCSx  ({0:d} dofs)".format(dofs)
            plt.plot(t_u, C_L, "-.", label=lbl, linewidth=2)
            y = turek[1:, 4]
            plt.plot(x, y, marker="x", markevery=50, ls="", ms=4, label=lbl)
            plt.title("Lift coefficient")
            plt.grid()
            plt.legend()
            plt.savefig(pth + "lift_comparison.png")
            plt.close()
        if (i % self.save_step) == 0:
            u = self.q_["u0"].vector().vec().array  # 10942
            v = self.q_["u1"].vector().vec().array
            p = self.q_["p"].vector().vec().array
            np.save(pth + "u_{:06d}.npy".format(i), u)
            np.save(pth + "v_{:06d}.npy".format(i), v)
            np.save(pth + "p_{:06d}.npy".format(i), p)
        return

    def theend_hook(self, SVD=False):
        print("post processing:")
        pth = self.pkg_dir + "results/" + self.simulation_start + "/"
        mkdir(pth)
        tmp_pth = self.temp_dir
        # save meshes as well as some other data
        V, Q = self.VV["u0"], self.VV["p"]
        np.save(pth + "drag.npy", self.C_D)
        np.save(pth + "lift.npy", self.C_L)
        np.save(pth + "drag.npy", V.mesh().cells())
        np.save(pth + "V_dof_coords.npy", V.tabulate_dof_coordinates())
        np.save(pth + "Q_dof_coords.npy", Q.tabulate_dof_coordinates())
        np.save(pth + "mesh_coords.npy", V.mesh().coordinates())
        np.save(pth + "mesh_cells.npy", V.mesh().cells())
        t = np.arange(0.0, self.T, self.dt) + self.dt
        np.save(pth + "time.npy", t)
        np.save(pth + "nu.npy", np.array([self.nu]))
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
        # move temp png files to results folder
        for quantity in ["p", "u", "v"]:
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
            for i, f in enumerate(onlyfiles):
                remove(tmp_pth + f)
        # move temp png files to results folder
        for f in listdir(tmp_pth):
            if isfile(join(tmp_pth, f)) and f.endswith(".png"):
                copy2(tmp_pth + f, pth + f)
                remove(tmp_pth + f)
        print("found", len(t), "timesteps and", len(onlyfiles), "saved files")
        return

    def plot(self):
        mesh = self.mesh
        u = self.q_["u0"].compute_vertex_values(mesh)
        v = self.q_["u1"].compute_vertex_values(mesh)
        p = self.q_["p"].compute_vertex_values(mesh)
        magnitude = (u ** 2 + v ** 2) ** 0.5
        x, y = mesh.coordinates().T
        tri = mesh.cells()
        fs = (12, 6)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=fs)
        ax1.quiver(x, y, u, v, magnitude)
        ax2.tricontourf(x, y, tri, p, levels=40)
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax1.set_title("velocity")
        ax2.set_title("pressure")
        return fig, (ax1, ax2)


if __name__ == "__main__":
    pkg_dir = dirname(__file__).split("oasisx")[0]
    path_to_config = pkg_dir + "/resources/cylinder/"
    config_file = path_to_config + "config.json"
    print(config_file)
    config = json.load(open(file=config_file, encoding="utf-8"))
    commandline_args = parse_command_line()

    my_domain = Cylinder(config)
    my_domain.set_parameters(commandline_args)
    my_domain.mesh_from_file()
    algorithm = FractionalStep(my_domain)
    my_domain.plot()
    algorithm.run()
    pth = pkg_dir + "results/" + my_domain.simulation_start + "/"
    copy2(config_file, pth + config_file.split("/")[-1])
