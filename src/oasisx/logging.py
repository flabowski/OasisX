#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 17:30:56 2022

@author: florianma
"""
import dolfin as df


def info_blue(s, check=True):
    BLUE = "\033[1;37;34m%s\033[0m"
    if df.MPI.rank(df.MPI.comm_world) == 0 and check:
        print(BLUE % s)


def info_green(s, check=True):
    GREEN = "\033[1;37;32m%s\033[0m"
    if df.MPI.rank(df.MPI.comm_world) == 0 and check:
        print(GREEN % s)


def info_red(s, check=True):
    RED = "\033[1;37;31m%s\033[0m"
    if df.MPI.rank(df.MPI.comm_world) == 0 and check:
        print(RED % s)
