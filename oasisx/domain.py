#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:52:29 2022

@author: florianma
"""
import dolfin as df
from oasisx.logging import info_green, info_red


class Domain:
    def __init__(self):
        #
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
        mesh = self.mesh
        return df.Constant((0,) * mesh.geometry().dim())

    def pre_solve_hook(self):
        raise NotImplementedError()

    def scalar_hook(self):
        raise NotImplementedError()

    def theend_hook(self):
        raise NotImplementedError()

    def mesh_from_file(self):
        self.mesh = df.Mesh()
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
        Cmax = 0.05
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

    def show_info(self, t, tstep, toc):
        msg = "Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}"
        info_green(msg.format(t, tstep, self.T))
        msg = "Total computing time on previous {0:d} timesteps = {1:f}"
        info_red(msg.format(self.print_intermediate_info, toc))
