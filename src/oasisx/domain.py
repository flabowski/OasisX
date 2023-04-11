#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:52:29 2022

@author: florianma
"""
import dolfin as df
from oasisx.logging import info_green, info_red
from oasisx.io import parse_command_line
from os.path import isfile, join, dirname, exists, expanduser
from os import mkdir
import numpy as np
from shutil import copy2
import json


class Domain:
    def __init__(self):
        return

    def get_problem_parameters(self):
        raise NotImplementedError()

    def scalar_source(self):
        return dict((ci, df.Constant(0)) for ci in self.scalar_components)

    def create_bcs(self):
        sys_comp = self.sys_comp
        return dict((ui, []) for ui in sys_comp)

    def initialize(self):
        raise NotImplementedError()

    def body_force(self):
        """Specify body force"""
        return NotImplementedError()

    def pre_solve_hook(self):
        raise NotImplementedError()

    def scalar_hook(self):
        raise NotImplementedError()

    def theend_hook(self):
        raise NotImplementedError()

    def mesh_from_file(self):
        self.mesh = df.Mesh()
        print(self.mesh_name)
        print(self.facet_name)
        with df.XDMFFile(self.mesh_name) as infile:
            infile.read(self.mesh)
        dim = self.mesh.topology().dim()
        mvc = df.MeshValueCollection("size_t", self.mesh, dim - 1)
        with df.XDMFFile(self.facet_name) as infile:
            infile.read(mvc, "name_to_read")
        self.mf = df.cpp.mesh.MeshFunctionSizet(self.mesh, mvc)
        self.ds_ = df.ds(subdomain_data=self.mf, domain=self.mesh)
        return

    def recommend_dt(self):
        Cmax = 0.5
        dt = Cmax * self.mesh.hmin() / self.Umean
        print("recommended dt =", dt)
        return dt

    def set_parameters(self, kwargs):
        # Update NS_namespace with all parameters modified through command line
        for key, val in kwargs.items():
            setattr(self, key, kwargs[key])
            if key not in self.__dict__.keys():
                raise KeyError("unknown key", key)
            elif isinstance(val, dict):
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, kwargs[key])
        return

    def load(self):
        pth = self.restart_folder
        print("loading from:", pth)
        for key, val in self.q_.items():
            val.vector().vec().array = np.load(pth + "q_" + key + ".npy")
        for key, val in self.q_1.items():
            val.vector().vec().array = np.load(pth + "q_1" + key + ".npy")
        for key, val in self.q_2.items():
            val.vector().vec().array = np.load(pth + "q_2" + key + ".npy")
        # TODO: load .json

    def save(self):
        # self = my_domain
        print("saving...")
        pth = self.pkg_dir + "checkpoints/" + self.simulation_start + "/"
        # pth_rel = "../checkpoints/" + self.simulation_start + "/"
        print(pth)
        mkdir(pth)
        for key, val in self.q_.items():
            ary = val.vector().vec().array
            np.save(pth + "q_" + key + ".npy", ary)
        for key, val in self.q_1.items():
            ary = val.vector().vec().array
            np.save(pth + "q_1" + key + ".npy", ary)
        for key, val in self.q_2.items():
            ary = val.vector().vec().array
            np.save(pth + "q_2" + key + ".npy", ary)
        mesh_dir = dirname(self.mesh_name) + "/"
        mesh_name = self.mesh_name.split("/")[-1].replace(".xdmf", "")
        copy2(mesh_dir + mesh_name + ".xdmf", pth + mesh_name + ".xdmf")
        copy2(mesh_dir + mesh_name + ".h5", pth + mesh_name + ".h5")

        facet_dir = dirname(self.facet_name) + "/"
        facet_name = self.facet_name.split("/")[-1].replace(".xdmf", "")
        copy2(facet_dir + facet_name + ".xdmf", pth + facet_name + ".xdmf")
        copy2(facet_dir + facet_name + ".h5", pth + facet_name + ".h5")

        params = self._to_dict()
        params["mesh_name"] = pth + mesh_name + ".xdmf"
        params["facet_name"] = pth + facet_name + ".xdmf"
        params["restart"] = True
        params["restart_folder"] = pth
        with open(pth + "config.json", "x") as outfile:
            json.dump(params, outfile, indent=2)

    def _to_dict(self):
        res = {}
        # print(self.getmembers())
        for k, v in self.__dict__.items():
            # print(k, type(v))
            if type(v) in [str, int, bool, float]:
                res[k] = v
            elif type(v) == dict:
                print(k, "is a dict, not supported yet")
        return res

    def show_info(self, t, tstep, toc):
        msg = "Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}"
        info_green(msg.format(t, tstep, self.T))
        msg = "Total computing time on previous {0:d} timesteps = {1:f}"
        info_red(msg.format(self.print_intermediate_info, toc))
