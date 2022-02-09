#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Runs a linear stability analysis of 2D Rayleigh Benard convection.
#
# Code written consulting Introduction to Modeling Convection in Planets and Stars
# by Gary Glatzmaier.
#
# TO-DO:
#     1. Write to file at the end of simulation to allow multiple simulations
#         running at once.
#     2. Add ability to read in initial conditions from a previous run.
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
import numpy as np
import math
import time
from datetime import datetime

import matplotlib.pyplot as plt

# Import my tridiagonal matrix solver from tridiagonal.py
from tridiagonal import tridiagonal

# Import problem parameters from params.py
import params


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
    return psi


# =====================
# =====CLA PARSING=====
# =====================
parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--test", help="Do not save output to log", action="store_true"
)
parser.add_argument(
    "-c", "--comment", help="Optional comment to add to log", default=""
)
parser.add_argument(
    "-l",
    "--logfile",
    help="Name of logfile to write to. Default=log.txt",
    default="log.txt",
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
parser.add_argument(
    "-i",
    "--initial",
    help="Input a filename to read initial conditions from. Default, will start from 0",
    nargs="?",
    default=False,
    const="inputs.py",
)
args = parser.parse_args()

if args.test:
    save_to_log = False
else:
    save_to_log = True
logfile = args.logfile
graphical = args.graphical

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
Nz = params.Nz  # Vertical resolution (Number of z-levels)
Nn = params.Nn  # Number of Fourier modes
Ra = params.Ra  # Rayleigh number (measures convective driving)
Pr = params.Pr  # Prandtl number (ratio of viscous to thermal diffusion)
a = params.a  # Aspect ratio of the box
nsteps = params.nsteps  # Total number of steps to run simulation for
output_z = math.floor(params.z_percent * Nz)  #
output_n = 1
dz = 1.0 / (Nz - 1)  # Spacing between adjacent z-levels
oodz2 = 1.0 / dz ** 2  # Constant for 1/dz^2 to avoid repeated recalculation
oo2dz = 1.0 / (2 * dz)
c = np.pi / a  # constant to avoid repeated calculation

# Initialise arrays for the problem variables.
psi = np.zeros(
    shape=(Nn, Nz)
)  # Velocity Streamfunction, psi[n][z] = nth mode of the zth level
omega = np.zeros(shape=(Nn, Nz))  # Vorticity, omega[n][z] = nth mode of the zth level
temp = np.zeros(shape=(Nn, Nz))  # Temperature, temp[n][z] = nth mode of the zth level

# Initialise arrays for the time derivatives of temperature and vorticity.
# arrays called as: var_dt[step][n][z]
# where step is current or previous, n is mode and z is z-level
omega_dt = np.zeros(shape=(2, Nn, Nz))  # time derivative of vorticity
temp_dt = np.zeros(shape=(2, Nn, Nz))  # time derivative of temperature

# Arrays for holding the amplitude of problem variables.
# Shape of 2 for previous and current timestep
t_amp = np.zeros(shape=(2, Nn))
omega_amp = np.zeros(shape=(2, Nn))
psi_amp = np.zeros(shape=(2, Nn))

# Array to hold the z-value for each z-level
z_vals = np.zeros(shape=(Nz), dtype=float)  # array to hold the height
for i in range(1, Nz + 1):
    z_vals[i - 1] = (i - 1) * dz

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

# Add to log.txt the parameters of this simulation run.
if save_to_log:
    with open(logfile, "a+") as log:
        log.write("====================\n\n")
        log.write("#" + dt_string + "\t" + args.comment + "\n")
        log.write("Nn = {}\t Nz = {}\t a = {}\n".format(Nn, Nz, a))
        log.write("Ra = {}\t Pr = {}\n".format(Ra, Pr))
        log.write("output z = {}\t output n = {}\n".format(output_z, output_n))
        log.write("dt = {:.3e}\titerations = {:.0e}\n".format(dt, nsteps))

# ====================
# ===INITIAL VALUES===
# ====================


if not (args.initial):
    # i.e. if starting from random initial conditions
    # Populate temperature array for initial temperatures (t=0)
    # Only need to initialise for n=0 and n=1
    for z in range(0, Nn):
        # when n=0:
        # T(z) = 1 - z for all z
        temp[0][z] = 1 - z_vals[z]
        # for n=1:
        # T_(n, z) = randbetween(-1, 1) * small constant * sin(pi*z) at t=0
        # Initial non-zero values represent small temp perturbations
        temp[1][z] = np.random.randint(-10, 10) / 10 * 0.001 * np.sin(np.pi * z_vals[z])
else:
    # Read in initial conditions from a previous run. NOT YET IMPLEMENTED
    print("Will read initial conditions from {}".format(args.initial))


# ====================
# ======SIM LOOP======
# ====================

if output_n == 0:
    print(
        "The output when n=0 is 0 for temperature amplitude is undefined for "
        + "\nVorticity and Psi"
    )
    print("Please input a value for output_n > 0")
    exit(1)
elif output_n > Nn:
    print("Please input a value for output_n that is <= Nn.")
    exit(2)

print("I\t| temperature \t| vorticity\t| streamfunction")
for iteration in range(0, int(nsteps + 1)):
    # print("Step No. {}".format(iteration))

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

    # ANALYSIS OUTPUT
    if iteration % 50 == 0:
        t_amp[current] = 0
        omega_amp[current] = 0
        psi_amp[current] = 0
        for n in range(Nn):
            t_amp[current][n] = temp[n][output_z]
            omega_amp[current][n] = omega[n][output_z]
            psi_amp[current][n] = psi[n][output_z]
        if iteration != 0:
            for n in range(1, Nn):
                temp_check[n] = np.log(np.abs(t_amp[current][n])) - np.log(
                    np.abs(t_amp[previous][n])
                )
                omega_check[n] = np.log(np.abs(omega_amp[current][n])) - np.log(
                    np.abs(omega_amp[previous][n])
                )
                psi_check[n] = np.log(np.abs(psi_amp[current][n])) - np.log(
                    np.abs(psi_amp[previous][n])
                )
            temp_amps = np.vstack((temp_amps, temp_check))
            omega_amps = np.vstack((omega_amps, omega_check))
            psi_amps = np.vstack((psi_amps, psi_check))
            print(
                "{}\t| {:.6f}\t| {:.6f}\t| {:.6f}".format(
                    iteration,
                    temp_check[output_n],
                    omega_check[output_n],
                    psi_check[output_n],
                )
            )

        t_amp[previous] = t_amp[current]
        omega_amp[previous] = omega_amp[current]
        psi_amp[previous] = psi_amp[current]

    if np.any(np.isnan(temp)):
        print("Temp is NaN. Exiting on iteration {}.".format(iteration))
        if save_to_log:
            print(
                "Sim ended early due to NaN temp. Ended on iteration {}".format(
                    iteration
                ),
                file=open(logfile, "a"),
            )
        break

if save_to_log:
    log = open(logfile, "a")
    log.write(
        "Temp {:.6}\t Omega {:.6}\t Psi {:.6}\n".format(
            temp_check[output_n], omega_check[output_n], psi_check[output_n]
        )
    )

end_t = time.time()
t_delta = end_t - start_t
print("Completed {} iterations in {:.2f} seconds.".format(iteration, t_delta))

for n in range(1, Nn):
    print("\nFor n={} mode:".format(n))
    print("Temperature check = {:.6f}".format(temp_amps[-1][n]))
    print("Vorticity check = {:.6f}".format(omega_amps[-1][n]))
    print("Streamfunction check = {:.6f}".format(psi_amps[-1][n]))

if save_to_log:
    log = open(logfile, "a")
    for n in range(1, Nn):
        log.write("\nFor n={} mode:\n".format(n))
        log.write("Temperature check = {:.6f}\n".format(temp_amps[-1][n]))
        log.write("Vorticity check = {:.6f}\n".format(omega_amps[-1][n]))
        log.write("Streamfunction check = {:.6f}\n".format(psi_amps[-1][n]))
    log.flush()
    log.close()

if graphical:
    xdata = np.arange(0, nsteps + 250, 250)
    ncols = 2
    nrows = (Nn - 1) // ncols + ((Nn - 1) % ncols > 0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(9, 3 * nrows), sharex=True)

    for n, ax in enumerate(fig.axes):
        if n >= Nn - 1:
            break
        ax.set_title("n={} mode".format(n + 1))
        ax.plot(xdata, temp_amps[:, n + 1], label="Temperature", color="r")
        ax.plot(xdata, omega_amps[:, n + 1], label="Vorticity", color="b")
        ax.plot(
            xdata,
            psi_amps[:, n + 1],
            label="Streamfunction",
            color="g",
            linestyle=":",
            linewidth=3,
        )
        ax.legend()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Amplitude")
        ax.annotate(
            "{:.6f}".format(temp_amps[-1][n + 1]),
            (xdata[-1], temp_amps[-1][n + 1]),
            xytext=(xdata[-1], temp_amps[-1][n + 1] - 0.5),
            ha="right",
            color="r",
        )
        ax.annotate(
            "{:.6f}".format(omega_amps[-1][n + 1]),
            (xdata[-1], omega_amps[-1][n + 1]),
            xytext=(xdata[-1], omega_amps[-1][n + 1] + 0.5),
            ha="right",
            color="b",
        )
        ax.annotate(
            "{:.6f}".format(psi_amps[-1][n + 1]),
            (xdata[-1], psi_amps[-1][n + 1]),
            xytext=(xdata[-1], psi_amps[-1][n + 1] + 1),
            ha="right",
            color="g",
        )
        ax.axhline(ls="--", color="k")

    fig.tight_layout()
    if args.savefig:
        plt.savefig(args.savefig)
    else:
        plt.show()

exit(0)