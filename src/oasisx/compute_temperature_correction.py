#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:07:17 2023

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def Psi(T):
    return 1 / 2 * (1 + erf((T - T_m) / (sigma * 2**0.5)))


tt = np.linspace(659, 661, 1000)
fig, ax = plt.subplots()
plt.plot(tt, Psi(tt))
plt.show()

T_m = 660
sigma = 0.1
c_p = 910.0
L = 396567.0
dT = 0.1  # T_-T_m
T_1 = np.array([660.0])
T_ = np.array([660 - dT])

fl_1 = np.array([0.5])


dH1 = L * fl_1
# H_1 = c_p * (T_1+273.15) + dH1
# H_ = c_p * (T_+273.15) + dH1
# dH_ = H_ - c_p*(T_m+273.15)
dH_ = dH1 + c_p * (T_ - T_m)
fl_ = dH_ / L
print(fl_)
fl_[fl_ < 0.0] = 0.0
fl_[fl_ > 1.0] = 1.0
print(fl_)
is_mushy = (0.0 < fl_) & (fl_ < 1.0)
T_[is_mushy] = T_m  # or some linear function
# fl_ = (H_ - c_p*(T_m+273.15)) / L
# fl_ = c_p*(T_-T_m)/L + fl_1
# 0.499770


# T_k = T_
# fl_k1 = f(T_k)
# fl_k2 = 1.5 * fl_1 - 0.5 * fl_2  # AB
# fl_k3 = fl_1 - dT / (L / c_p)
# fig, ax = plt.subplots()
# for fl_k, c in zip([fl_k1, fl_k2, fl_k3], ["r", "g", "b"]):
#     # L/c_p
#     for k in range(10):
#         alpha = dT / (L / c_p)
#         dH = L * fl_k

#         dH_1 = L * fl_1
#         dH_k = L * fl_k
#         d_dH = dH_1 - dH_k  # change in
#         temperature_correction = d_dH / c_p
#         print(temperature_correction)  # 15...150
#         T_k = T_ + temperature_correction * alpha
#         fl_k = f(T_k)
#         plt.plot(k, fl_k, c + "o")
# plt.show()
