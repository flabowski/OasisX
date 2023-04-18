#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:52:29 2022

@author: florianma
"""
import dolfin as df
from oasisx.logging import info_green, info_red
from oasisx.io import parse_command_line
from os.path import isfile, join, dirname, exists, expanduser, abspath, isdir
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

    def recommend_dt(self):
        Cmax = 0.5
        dt = Cmax * self.mesh.hmin() / self.Umean
        print("recommended dt =", dt)
        return dt

    def set_parameters(self, kwargs):
        raise ValueError("please dont do that. use the config instead")
        return

    def load(self):
        pth = "/".join(self.config["origin"].split("/")[:-1]) + "/"
        print("loading from:", pth)
        for key, val in self.q_.items():
            val.vector().vec().array = np.load(pth + "q_" + key + ".npy")
        for key, val in self.q_1.items():
            val.vector().vec().array = np.load(pth + "q_1" + key + ".npy")
        for key, val in self.q_2.items():
            val.vector().vec().array = np.load(pth + "q_2" + key + ".npy")

    def save(self, tstep):
        print("saving...")
        pth = self.temp_dir + "checkpoint_{:.0f}/".format(tstep)
        print(pth)
        if not isdir(pth):
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
        origin = "/".join(self.config["origin"].split("/")[:-1]) + "/"
        copy2(origin + "mesh.xdmf", pth + "mesh.xdmf")
        copy2(origin + "mesh.h5", pth + "mesh.h5")
        copy2(origin + "mf.xdmf", pth + "mf.xdmf")
        copy2(origin + "mf.h5", pth + "mf.h5")

        params = self.config
        params["restart"] = True
        with open(pth + "config.json", "x") as outfile:
            json.dump(params, outfile, indent=4)

    def show_info(self, t, tstep, toc):
        msg = "Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}"
        info_green(msg.format(t, tstep, self.config["T"]))
        msg = "Total computing time on previous {0:d} timesteps = {1:f}"
        info_red(msg.format(self.config["print_intermediate_info"], toc))
