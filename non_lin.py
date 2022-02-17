#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Runs a linear stability analysis of 2D Rayleigh Benard convection.
#
# Code written consulting Introduction to Modeling Convection in Planets and Stars
# by Gary Glatzmaier.
#
# KNOWN BUGS:
#     - Reading initial conditions from same folder as output logs will over-write old output logs
#
# TO-DO:
#   0.5. Email Simon and ask if he knows what to do RE the numerical tests
#     1. Run tests outlined in Glatzmaier §4.4 to verify accuracy and validity.
#     2. Write a process.py to perform post-processing on the output.h5 file.
#
# usage: lin_stab.py [-h] [-t] [-c COMMENT] [-l LOGFILE] [-g] [-s]
#
# optional arguments:
#   -h, --help            show this help message and exit
#   -t, --test            Do not save output to log
#   -c 'COMMENT', --comment 'COMMENT'
#                         Optional comment to add to log
#   -l 'LOGFILE', --logfile 'LOGFILE'
#                         Name of logfile to write to. Default=log.txt
#   -g, --graphical       Plots the amplitude of n-modes against iteration
#                         number
#   -s, --savefig         Will save the figure as out.png
"""
#
# ===================
# ======IMPORTS======
# ===================
import argparse
import os
import numpy as np
import math
import time  # For timing the simulation
from datetime import datetime  # For printing the date and time in the log
import h5py as h5  # For analysis data output

import matplotlib.pyplot as plt

# Import my tridiagonal matrix solver from tridiagonal.py
from tridiagonal import tridiagonal

# Import problem parameters from params.py
import params

# ====================
# FUNCTION DEFINITIONS
# ====================
def calculate_dt(psi, Nn, Nx, Nz, x_vals, c, dz, dt):
    """
    Checks that the timestep satisfies the CFL condition and the diffusion limit, and updates it accordingly

    Inputs:
        psi - array containing the velocity streamfunction
        Nn - integer containing the number of N nodes
        Nx - integer containing the number of x-levels
        Nz - integer value of the number of z-levels
        c - constant representing pi/a where a is the aspect ratio of sim box
        dz - distance between z-levels
        dt - timestep to be checked
    Returns:
        dt - new timestep
    """
    v_z = np.zeros((Nz))
    for k in range(0, Nz):
        for n in range(0, Nn):
            v_z[k] += (n * c) * psi[n][k]
    CFL_dt = dz / np.max(np.abs(np.ndarray.flatten(v_z)))
    diff_dt = (dz ** 2) / 4
    if diff_dt <= CFL_dt and dt < diff_dt:
        # diffusion timestep dominates and dt doesn't need to change.
        return dt
    elif diff_dt <= CFL_dt and dt >= diff_dt:
        if params.Pr <= 1:
            dt = 0.9 * ((dz * dz) / 4)  # 90% of Constaint 2.19 for safety.
        else:  # if Prandtl > 1, divide 2.19 by Pr to avoid numerical instability
            dt = 0.9 * ((dz * dz) / (4 * params.Pr))
        return dt
    elif CFL_dt < diff_dt:
        if 1.2 * dt >= CFL_dt:
            dt = 0.9 * CFL_dt
        elif CFL_dt >= 5 * dt:
            dt = 4 * CFL_dt
        else:
            dt = dt
        return dt


def update_derivatives(
    temp_dt, omega_dt, psi, Nn, Nz, c, oodz2, oo2dz, temp, Ra, Pr, omega
):
    """
	Updates the time-derivatives of temp_dt and omega_dt using the Vertical Finite-Difference method to approximate spatial double derivates

	Inputs:
		temp_dt - array containing the time derivative of temperature for all n, z at the current and previous timestep
		omega_dt - array containing the time derivative of temperature for all n, z at the current and previous timestep
		psi - array containing the velocity streamfunction for all n and z
		Nn - array containing the number of N nodes, i.e. the horizontal resolution
		Nz - array containing the number of z levels i.e. the vertical resolution
		c - constant containing the value of pi/a where a is the aspect ratio of the simulation box
		oodz2 - constant containing 1/(dz)^2 where dz is the distance between z levels.
		temp - array containing the temperature for all n and z
		Ra - constant containing the inputted Rayleigh number
		Pr - constant containing the inputted Prandtl number
		omega - array containing the vorticity for all n and z
	Returns:
		temp_dt - as above but with new values for "current step"
		omega_dt - as above but with new values for "current step"
	"""
    current = 1
    for z in range(1, Nz - 1):
        # For all linear terms:
        # n = 0
        temp_dt[current][0][z] = oodz2 * (
            temp[0][z + 1] - 2 * temp[0][z] + temp[0][z - 1]
        )
        for n in range(1, Nn):  # 1 <= n <= Nn
            temp_dt[current][n][z] = (
                oodz2 * (temp[n][z + 1] - 2 * temp[n][z] + temp[n][z - 1])
            ) - ((n * c) ** 2) * temp[n][z]
            omega_dt[current][n][z] = Ra * Pr * (n * c) * temp[n][z] + Pr * (
                oodz2 * (omega[n][z + 1] - 2 * omega[n][z] + omega[n][z - 1])
                - (n * c) ** 2 * omega[n][z]
            )
        # for all non-linear terms:
        # n = 0
        for n_p in range(1, Nn):
            temp_dt[current][0][z] += (
                -(n_p * c / 2)
                * oo2dz
                * (
                    temp[n_p][z] * (psi[n_p][z + 1] - psi[n_p][z - 1])
                    + psi[n_p][z] * (temp[n_p][z + 1] - temp[n_p][z - 1])
                )
            )
        # n from 1 to Nn
        for n in range(1, Nn):
            # n' = 0:
            temp_dt[current][n][z] += (
                -(n * c * oo2dz) * psi[n][z] * (temp[0][z + 1] - temp[0][z - 1])
            )
            for n_p in range(1, Nn):
                # Kronecker-Delta contributions from temp_dt and omega_dt
                n_pp = n - n_p
                if (n_pp >= 1) and (n_pp <= Nn - 1):
                    temp_dt[current][n][z] += (-0.5 * c) * (
                        (
                            -n_p
                            * oo2dz
                            * (psi[n_pp][z + 1] - psi[n_pp][z - 1])
                            * temp[n_p][z]
                        )
                        + (
                            n_pp
                            * oo2dz
                            * (temp[n_p][z + 1] - temp[n_p][z - 1])
                            * psi[n_pp][z]
                        )
                    )

                    omega_dt[current][n][z] += (-0.5 * c) * (
                        (
                            -n_p
                            * oo2dz
                            * (psi[n_pp][z + 1] - psi[n_pp][z - 1])
                            * omega[n_pp][z]
                        )
                        + (
                            n_pp
                            * oo2dz
                            * (omega[n_p][z + 1] - omega[n_p][z - 1])
                            * psi[n_pp][z]
                        )
                    )
                n_pp = n + n_p  # When n'' = n + n'. then delta_n''-n, n = 1
                if (n_pp >= 1) and (n_pp <= Nn - 1):
                    temp_dt[current][n][z] += (-0.5 * c) * (
                        (
                            n_p
                            * oo2dz
                            * (psi[n_pp][z + 1] - psi[n_pp][z - 1])
                            * temp[n_p][z]
                        )
                        + (
                            n_pp
                            * oo2dz
                            * (temp[n_p][z + 1] - temp[n_p][z - 1])
                            * psi[n_pp][z]
                        )
                    )
                    omega_dt[current][n][z] += (
                        (-0.5 * c)
                        * -1
                        * (
                            (
                                n_p
                                * oo2dz
                                * (psi[n_pp][z + 1] - psi[n_pp][z - 1])
                                * omega[n_p][z]
                            )
                            + (
                                n_pp
                                * psi[n_pp][z]
                                * oo2dz
                                * (omega[n_p][z + 1] - omega[n_p][z - 1])
                            )
                        )
                    )
                n_pp = n_p - n
                if (n_pp >= 1) and (n_pp <= Nn - 1):
                    temp_dt[current][n][z] += (-0.5 * c) * (
                        (
                            n_p
                            * oo2dz
                            * (psi[n_pp][z + 1] - psi[n_pp][z - 1])
                            * temp[n_p][z]
                        )
                        + (
                            n_pp
                            * oo2dz
                            * (temp[n_p][z + 1] - temp[n_p][z - 1])
                            * psi[n_pp][z]
                        )
                    )
                    omega_dt[current][n][z] += (-0.5 * c) * (
                        (
                            n_p
                            * oo2dz
                            * (psi[n_pp][z + 1] - psi[n_pp][z - 1])
                            * omega[n_p][z]
                        )
                        + (
                            n_pp
                            * oo2dz
                            * (omega[n_p][z + 1] - omega[n_p][z - 1])
                            * psi[n_pp][z]
                        )
                    )

    return temp_dt, omega_dt


def adams_bashforth(temp, omega, temp_dt, omega_dt, dt, Nn, Nz):
    """
	Uses the Adams-Bashforth method to increase temperature and vorticity for each timestep.

	Inputs:
		temp - temperature array
		omega - vorticity array
		temp_dt - 3D array for time-derivative of temperature
		omega_dt - 3D array for time-derivative of vorticity
		dt - timestep
		Nn - number of horizontal modes
		Nz - number of vertical z-levels
	Returns:
		temp, omega
	"""
    for n in range(0, Nn):
        for z in range(0, Nz):
            # Adams-Bashforth Time Integration: T_t+dt = Tt + dt/2 * (3*dT/dt_t - dT/dt_t-dt)
            temp[n][z] = temp[n][z] + (dt / 2) * (
                3 * temp_dt[current][n][z] - temp_dt[previous][n][z]
            )
            omega[n][z] = omega[n][z] + (dt / 2) * (
                3 * omega_dt[current][n][z] - omega_dt[previous][n][z]
            )
        # print("for n={}:\ntemp={}\nomega={}".format(n, temp[n], omega[n]))
    return temp, omega


def update_streamfunction(psi, sub, dia, sup, omega, Nn, Nz, c, oodz2):

    # First - construct the diagonal matrix
    for n in range(0, Nn):
        for z in range(0, Nz):
            if z == 0:
                dia[z] = 1
            else:
                dia[z] = (n * c) ** 2 + 2 * oodz2
        dia[-1] = 1
        psi[n] = tridiagonal(sub, dia, sup, omega[n])
        # print("for n={}:\npsi={}".format(n, psi[n]))
    return psi


def output_maximums(temp, omega, psi, Nn, Nz):
    max_T = np.max(np.ndarray.flatten(temp)[1:])
    index_mT = np.where(temp == max_T)
    max_omega = np.max(np.ndarray.flatten(omega))
    index_mO = np.where(omega == max_omega)
    max_psi = np.max(np.ndarray.flatten(psi))
    index_mP = np.where(psi == max_psi)
    print("Temp\t| {:.6e}\t| {}\t| {}".format(max_T, index_mT[0], index_mT[1]))
    print("Omega\t| {:.6e}\t| {}\t| {}".format(max_omega, index_mO[0], index_mO[1]))
    print("Psi\t| {:.6e}\t| {}\t| {}".format(max_psi, index_mP[0], index_mP[1]))


# =====================
# =====CLA PARSING=====
# =====================
parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--test", help="Do not save output to log", action="store_true"
)
parser.add_argument(
    "-o",
    "--output_folder",
    help="Name of folder to store output data in. Default=output/",
    default="output/",
)
parser.add_argument(
    "-c", "--comment", help="Optional comment to add to log", default=""
)
parser.add_argument(
    "-i", "--initial", help="Input a path to a file to read initial conditions from",
)
parser.add_argument(
    "-g",
    "--graphical",
    help="Plots the amplitude of n-modes against iteration number",
    action="store_true",
)
parser.add_argument(
    "-s",
    "--savefig",
    help="Will save the figure with provided filename. Default=out.png",
    nargs="?",
    default=False,
    const=True,
)
args = parser.parse_args()

if args.test:
    save_output = False
else:
    save_output = True

if save_output:
    if "/" in args.output_folder[-1]:
        outpath = os.path.normpath(args.output_folder)
    else:
        outpath = os.path.normpath(args.output_folder + "/")
    os.makedirs(outpath, exist_ok=True)
    logfile = outpath + "/log.txt"

graphical = args.graphical
if args.initial:
    if not os.path.isfile(args.initial):
        print("{} is not a valid file.".format(args.initial))
        exit(3)

# ====================
# ===INITIALISATION===
# ====================
run_begin = datetime.now()
dt_string = run_begin.strftime("%d/%m/%Y %H:%M:%S")
# Begin timing
start_t = time.time()

# Allows for extra readability when calling amplitude values
current = 1
previous = 0

# Set up parameters
a = params.a  # Aspect ratio of the box
if not (args.initial):
    Nz = params.Nz  # Vertical resolution (Number of z-levels)
    Nx = a * Nz  # Number of x-levels (horizontal resolution)
    Nn = params.Nn  # Number of Fourier modes
else:
    # TODO: read in Nz, Nx, Nn from initial file.
    """"""
    restart_vals = np.load(args.initial)
    Nn, Nz, Nx = restart_vals["simparams"][:3]
    Nn, Nz, Nx = map(int, (Nn, Nz, Nx))
Ra = params.Ra  # Rayleigh number (measures convective driving)
Pr = params.Pr  # Prandtl number (ratio of viscous to thermal diffusion)
dz = 1.0 / (Nz - 1)  # Spacing between adjacent z-levels
oodz2 = 1.0 / dz ** 2  # Constant for 1/dz^2 to avoid repeated recalculation
oo2dz = 1.0 / (2 * dz)
c = np.pi / a  # constant to avoid repeated calculation

# Iteration Parameters
nsteps = params.nsteps  # Total number of steps to run simulation for
iprint = 50
post_print = 10
timestep_check = 200
write_restart = 100

# Initialise arrays for the problem variables.
# Velocity Streamfunction, psi[n][z] = nth mode of the zth level
psi = np.zeros(shape=(Nn, Nz))
# Vorticity, omega[n][z] = nth mode of the zth level
omega = np.zeros(shape=(Nn, Nz))
# Temperature, temp[n][z] = nth mode of the zth level
temp = np.zeros(shape=(Nn, Nz))

# Initialise arrays for the time derivatives of temperature and vorticity.
# arrays called as: var_dt[step][n][z]
# where step is current or previous, n is mode and z is z-level
temp_dt = np.zeros(shape=(2, Nn, Nz))  # time derivative of temperature
omega_dt = np.zeros(shape=(2, Nn, Nz))  # time derivative of vorticity

if args.initial:
    # TODO: read psi, omega, temp and previous ts of omega_dt and temp_dt from file
    """"""
    psi = restart_vals["psi"]
    omega = restart_vals["omega"]
    temp = restart_vals["temp"]
    temp_dt[0] = restart_vals["prev_dtempdt"]
    omega_dt[0] = restart_vals["prev_domgdt"]

# Arrays for holding the amplitude of problem variables.
# Shape of 2 for previous and current timestep
t_amp = np.zeros(shape=(2, Nn))
omega_amp = np.zeros(shape=(2, Nn))
psi_amp = np.zeros(shape=(2, Nn))

# Array to hold the z-value for each z-level
z_vals = np.zeros(shape=(Nz), dtype=float)  # array to hold the height
for i in range(1, Nz + 1):
    z_vals[i - 1] = (i - 1) * dz

x_vals = np.zeros(shape=(Nx), dtype=float)
for i in range(1, Nx + 1):
    x_vals[i - 1] = a * (i - 1) / (Nx - 1)

# Arrays to hold the output values for the analysis step
temp_check = np.zeros(shape=(Nn))
omega_check = np.zeros(shape=(Nn))
psi_check = np.zeros(shape=(Nn))
temp_amps = np.zeros(shape=(Nn))
omega_amps = np.zeros(shape=(Nn))
psi_amps = np.zeros(shape=(Nn))

# ====================
# =TRIDIAGONAL SET-UP=
# ====================

# Matrix defined by Eq 2.21 in Glatzmaier
# Arrays to hold the values of the tridiagonal matrix.
sub = []
dia = np.zeros(shape=(Nz))
sup = []

# Super- and sub-diagonal arrays do not change with timestep or mode, so can be
# populated now.

# Populate the subdiagional array. final element=0, all others = -(1/dz^2)
for i in range(1, Nz - 1):
    sub.append(-oodz2)
sub.append(0)
# Populate the superdiagonal array. First element=0, all others = -(1/dz^2)
sup.append(0)
for i in range(1, Nz - 1):
    sup.append(-oodz2)

# Calculate the timestep. Constraint 2.19 is dt < (dz^2)/4 to avoid numerical instability
if Pr <= 1:
    dt = 0.9 * ((dz * dz) / 4)  # 90% of Constaint 2.19 for safety.
else:  # if Prandtl > 1, divide 2.19 by Pr to avoid numerical instability
    dt = 0.9 * ((dz * dz) / (4 * Pr))

dt = 3e-6

print("dt = {}".format(dt))
# Add to log.txt the parameters of this simulation run.
if save_output:
    with open(logfile, "a+") as log:
        log.write("====================\n\n")
        log.write("#" + dt_string + "\t" + args.comment + "\n")
        log.write("Nn = {}\t Nz = {}\t a = {}\n".format(Nn, Nz, a))
        log.write("Ra = {}\t Pr = {}\n".format(Ra, Pr))
        log.write("dt = {:.3e}\titerations = {:.0e}\n".format(dt, nsteps))
    if not (args.initial):
        hf = h5.File(outpath + "/output.h5", "w", track_order=True)
        temp_group = hf.create_group("variables/temperature")
        vort_group = hf.create_group("variables/vorticity")
        stream_group = hf.create_group("variables/streamfunction")
    else:
        hf = h5.File(outpath + "/output.h5", "a", track_order=True)
    simparams = [Nn, Nz, Nx, 0, 0]
# ====================
# ===INITIAL VALUES===
# ====================


if not (args.initial):
    # i.e. if starting from random initial conditions
    # Populate temperature array for initial temperatures (t=0)
    # Only need to initialise for n=0 and n=1
    for z in range(0, Nz):
        # when n=0:
        # T(z) = 1 - z for all z
        temp[0][z] = 1 - z_vals[z]
        # for n=1:
        # T_(n, z) = randbetween(-1, 1) * small constant * sin(pi*z) at t=0
        # Initial non-zero values represent small temp perturbations
        temp[1][z] = 0.01 * np.sin(np.pi * z_vals[z])
    temp[1][Nz - 1] = 0  # to avoid the floating point rounding error for sin(pi)!=0
    curr_time = 0
    iter_start = 0
else:
    # Read in curr_time and iter_start from a previous run. NOT YET IMPLEMENTED
    curr_time = restart_vals["simparams"][3]
    iter_start = int(restart_vals["simparams"][4])


# ====================
# ======SIM LOOP======
# ====================

print("var\t| max value \t| n\t| z")
for iteration in range(iter_start, int(iter_start + nsteps + 1)):
    print("{}".format(iteration), end="\r")

    # Update derivatives of temperature and vorticity for the current timestep
    temp_dt, omega_dt = update_derivatives(
        temp_dt, omega_dt, psi, Nn, Nz, c, oodz2, oo2dz, temp, Ra, Pr, omega
    )
    # print(list(temp_dt))
    # Update temperature and vorticity using Adams-Bashforth Time Integration
    temp, omega = adams_bashforth(temp, omega, temp_dt, omega_dt, dt, Nn, Nz)
    # Update velocity streamfunction by creating tridiagonal matrix and solving
    psi = update_streamfunction(psi, sub, dia, sup, omega, Nn, Nz, c, oodz2)

    for n in range(0, Nn):
        for z in range(0, Nz):
            temp_dt[previous][n][z] = temp_dt[current][n][z]
            omega_dt[previous][n][z] = omega_dt[current][n][z]

    if iteration % write_restart == 0 and iteration != 0 and save_output:
        simparams[3] = curr_time
        simparams[4] = iteration
        np.savez_compressed(
            outpath + "/restart",
            simparams=simparams,
            temp=temp,
            omega=omega,
            psi=psi,
            prev_dtempdt=temp_dt[0],
            prev_domgdt=omega_dt[0],
        )

    # ANALYSIS OUTPUT
    if iteration % iprint == 0 and iteration != 0:
        print("i={}\tdt={}".format(iteration, dt))
        output_maximums(temp, omega, psi, Nn, Nz)
    if iteration % post_print == 0 and save_output:
        iter_temp = temp_group.create_dataset(str(iteration), data=temp)
        iter_vort = vort_group.create_dataset(str(iteration), data=omega)
        iter_stream = stream_group.create_dataset(str(iteration), data=psi)

    if np.any(np.isnan(temp)):
        print("Temp is NaN. Exiting on iteration {}.".format(iteration))
        if save_output:
            print(
                "Sim ended early due to NaN temp. Ended on iteration {}".format(
                    iteration
                ),
                file=open(logfile, "a"),
            )
        break
    if iteration % timestep_check == 0:
        dt = calculate_dt(psi, Nn, Nx, Nz, x_vals, c, dz, dt)

    curr_time += dt

hf.close()

end_t = time.time()
t_delta = end_t - start_t
print("Completed {} iterations in {:.2f} seconds.".format(iteration, t_delta))

exit(0)
