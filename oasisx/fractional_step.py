#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:45:42 2022

@author: florianma
"""
import dolfin as df
import numpy as np
from oasisx.logging import info_red
from oasisx.utilities import OasisTimer, getMemoryUsage, OasisMemoryUsage
import oasisx.ipcs_abcn as solver
import matplotlib.pyplot as plt
from shutil import copy2
import matplotlib

# import json


class FractionalStep:
    def __init__(self, domain):
        np.set_printoptions(suppress=True)
        matplotlib.use("Agg")  # Qt5Agg for normal interactive, Agg for offscreen
        plt.close("all")
        self.initial_memory_use = getMemoryUsage()
        self.oasis_memory = OasisMemoryUsage("Start")

        domain.list_problem_components()
        domain.declare_components()
        domain.initialize_components()
        domain.create_bcs()
        # TODO: Read in previous solution if restarting
        # TODO: LES setup
        # TODO: Non-Newtonian setup.
        domain.apply_bcs()
        self.domain = domain

    def run(self):
        # shortcuts
        max_iter = self.domain.max_iter
        max_error = self.domain.max_error
        # use_ROM = self.domain.use_ROM
        # pressure_step = self.domain.pressure_step
        # velocity_correction_step = self.domain.velocity_correction_step
        debug = self.domain.debug
        decompose_results = self.domain.decompose_results
        monitor_convergence = self.domain.krylov_solvers["monitor_convergence"]

        # if use_ROM:
        #     dir_oasis = "/home/florianma@ad.ife.no/ad_disk/Florian/Repositoties/Oasis/"
        #     dir_rom = dir_oasis + "results/cylinder_reference/"

        #     U = np.load(dir_rom + "ROM_U_p.npy")
        #     S = np.load(dir_rom + "ROM_S_p.npy")
        #     VT = np.load(dir_rom + "ROM_VT_p.npy")
        #     X_min = np.load(dir_rom + "ROM_X_min_p.npy")
        #     X_range = np.load(dir_rom + "ROM_X_range_p.npy")
        #     X = np.load(dir_rom + "X_p.npy")
        #     time = np.load(dir_rom + "time.npy")
        #     nu = np.load(dir_rom + "nu.npy")
        #     grid = (time,)
        #     r = np.sum(np.cumsum(S) / np.sum(S) < 0.9999)  # -0.00373 ... 0.00450
        #     ROM = ReducedOrderModel(grid, U[:, :r], S[:r], VT[:r], X_min, X_range)

        tx = OasisTimer("Timestep timer")
        tx.start()
        total_timer = OasisTimer("Start simulations", True)

        print_info = self.domain.use_krylov_solvers and monitor_convergence
        it0 = self.domain.iters_on_first_timestep
        max_iter = self.domain.max_iter

        fit = solver.FirstInner(self.domain)
        tvs = solver.TentativeVelocityStep(self.domain)
        prs = solver.PressureStep(self.domain)
        for ci in self.domain.scalar_components:
            scs = solver.ScalarSolver(self.domain)
        stop = False
        t = 0.0
        tstep = 0
        total_inner_iterations = 0
        while (t - df.DOLFIN_EPS) < (self.domain.T - self.domain.dt) and not stop:

            # if tstep == 30:
            #     stop = True

            # t += self.domain.dt  # avoid annoying rounding errors
            t = np.round(t + self.domain.dt, decimals=8)
            tstep += 1

            inner_iter = 0
            num_iter = max(it0, max_iter) if tstep <= 10 else max_iter

            ts = OasisTimer("start_timestep_hook")
            self.domain.start_timestep_hook(t)  # update bcs
            ts.stop()

            tr0 = OasisTimer("ROM")
            # if use_ROM:
            #     if tstep > 1:
            #         offset = 0.0  # -0.5 * self.domain.dt
            #     else:
            #         offset = 0.0
            #     guess1 = self.domain.q_["p"].vector().vec().array.copy()
            #     guess2 = ROM.predict([t - offset])[0].ravel()
            #     guess3 = X[:, tstep - 1]  # only works if dt is the same..
            #     self.domain.q_["p"].vector().vec().array = guess2.copy()
            tr0.stop()
            udiff = 1e8
            while udiff > max_error and inner_iter < num_iter:
                inner_iter += 1
                total_inner_iterations += 1

                t0 = OasisTimer("Tentative velocity")
                if inner_iter == 1:
                    # lesmodel.les_update()
                    # nnmodel.nn_update()
                    fit.assemble()
                    tvs.A = fit.A
                udiff = 0
                for i, ui in enumerate(self.domain.u_components):
                    t1 = OasisTimer("Solving tentative velocity " + ui, print_info)
                    tvs.assemble(ui=ui)  # uses p_ to compute gradp
                    udiff = tvs.solve(ui=ui, udiff=udiff)
                    self.domain.velocity_tentative_hook(ui=ui)
                    t1.stop()
                t0.stop()

                t2 = OasisTimer("Pressure solve", print_info)
                # if tstep % pressure_step == 0 or tstep <= 10:
                prs.assemble()
                pdiff = prs.solve()
                self.domain.pressure_hook()
                t2.stop()
                # discard the extra inner iterations of the first 10 outer itartions
                if inner_iter < max_iter:
                    if hasattr(self.domain, "udiff"):
                        self.domain.udiff[tstep - 1, inner_iter - 1] = udiff
                    if hasattr(self.domain, "pdiff"):
                        self.domain.pdiff[tstep - 1, inner_iter - 1] = pdiff

                if debug:
                    p = self.domain.q_["p"].vector().vec().array
                    p_ = self.domain.q_["p_"].vector().vec().array
                    e3 = np.abs(p - p_)
                    phi = np.abs(self.domain.dp_.vector().vec().array)
                    print(
                        inner_iter,
                        "{:.6f}\t{:.8f}\t{:.8f}".format(udiff, phi.max(), e3.max()),
                    )
                self.domain.print_velocity_pressure_info(num_iter, inner_iter, udiff)
            if debug:
                print(
                    "step {:.0f}, time: {:.6f} s. Inner loop stopped after {:.0f}"
                    " inner iterations. Total inner iterations: {:.0f}".format(
                        tstep, t, inner_iter, total_inner_iterations
                    )
                )
            # if use_ROM:
            #     p = self.domain.q_["p"].vector().vec().array
            #     e1 = np.abs(p - guess1)
            #     e2 = np.abs(p - guess2)
            #     e3 = np.abs(p - guess3)
            #     print(
            #         "{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t".format(
            #             e1.max(), e1.mean(), e2.max(), e2.mean(), e3.max(), e3.mean()
            #         )
            #     )
            # Update velocity
            t3 = OasisTimer("Velocity update")
            # if tstep % velocity_correction_step == 0 or tstep <= 10:
            for i, ui in enumerate(self.domain.u_components):
                tvs.velocity_update(ui=ui)
            t3.stop()

            # Scalar solver
            if len(self.domain.scalar_components) > 0:
                t3 = OasisTimer("Solving scalar {}".format(ci))  # print_solve_info
                scs.assemble()
                for ci in self.domain.scalar_components:
                    scs.solve(ci)
                    self.domain.scalar_hook()
                t3.stop()

            t4 = OasisTimer("temporal hook")
            self.domain.temporal_hook(t=t, tstep=tstep, ps=prs, tvs=tvs)
            t4.stop()

            # TODO: Save solution if required and check for killoasis file
            # stop = io.save_solution()

            ts = OasisTimer("advance / assemble bc")
            self.domain.advance()
            ts.stop()

            # Print some information
            if tstep % self.domain.print_intermediate_info == 0:
                toc = tx.stop()
                self.domain.show_info(t, tstep, toc)
                df.list_timings(df.TimingClear.clear, [df.TimingType.wall])
                tx.start()

            # AB projection for pressure on next timestep
            if (
                self.domain.AB_projection_pressure
                and t < (self.domain.T - tstep * df.DOLFIN_EPS)
                and not stop
            ):
                self.domain.q_["p"].vector().axpy(0.5, self.domain.dp_.vector())

        total_timer.stop()
        df.list_timings(df.TimingClear.keep, [df.TimingType.wall])
        info_red("Total computing time = {0:f}".format(total_timer.elapsed()[0]))
        self.oasis_memory("Final memory use ")
        # total_initial_dolfin_memory
        m = df.MPI.sum(df.MPI.comm_world, self.initial_memory_use)
        info_red("Memory use for importing dolfin = {} MB (RSS)".format(m))
        info_red(
            "Total memory use of solver = {:.4f} MB (RSS)".format(
                self.oasis_memory.memory - m
            )
        )
        # Final hook
        self.domain.theend_hook()
        # TODO: save data that was actually used. DO that in the end hook
