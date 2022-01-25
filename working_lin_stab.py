#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs a linear stability analysis of 2D Rayleigh Benard convection.

Code written consulting Introduction to Modeling Convection in Planets and Stars
by Gary Glatzmaier.
"""
"""
===================
======IMPORTS======
===================
"""
import argparse
import numpy as np
import math
import time
from datetime import datetime

import matplotlib.pyplot as plt

#Import my tridiagonal matrix solver from tridiagonal.py
from tridiagonal import tridiagonal

#Import problem parameters from params.py
import params

def finite_diff(temp_dt, omega_dt, psi, Nn, Nz, c, oodz2, temp, Ra, Pr, omega):
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
	for n in range(1, Nn):
		for z in range(1, Nz-1):
			# Vertical Finite-Difference approximation for double-derivatives
			temp_dt[current][n][z] = ((n * c)*psi[n][z]*(oodz2*(temp[n][z+1] - 2*temp[n][z] + temp[n][z-1])) - ((n*c)**2) * temp[n][z])
			omega_dt[current][n][z] = Ra*Pr*(n*c)*temp[n][z] + Pr*(oodz2*(omega[n][z+1] - 2*omega[n][z] + omega[n][z-1]) - (n * c)**2 * omega[n][z])
	return temp_dt, omega_dt

def adams_bashford(temp, omega, temp_dt, omega_dt, dt, Nn, Nz):
	"""
	Uses the Adams-Bashford method to increase temperature and vorticity for each timestep.

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
			#Adams-Bashford Time Integration: T_t+dt = Tt + dt/2 * (3*dT/dt_t - dT/dt_t-dt)
			temp[n][z] = temp[n][z] + (dt/2)*(3*temp_dt[current][n][z] - temp_dt[previous][n][z])
			omega[n][z] = omega[n][z] + (dt/2) * (3*omega_dt[current][n][z] - omega_dt[previous][n][z])
	return temp, omega

def update_streamfunction(psi, sub, dia, sup, omega, Nn, Nz, c, oodz2):
	for n in range(0, Nn):
		for z in range(0, Nz):
			if z==0:
				dia[z] = 1
			else:
				dia[z] = ((n*c)**2 + 2*oodz2)
		dia[-1] = 1
		psi[n] = tridiagonal(sub, dia, sup, omega[n])
	return psi
"""
=====================
=====CLA PARSING=====
=====================
"""
parser = argparse.ArgumentParser()
parser.add_argument('-t', "--test", help='Do not save output to log', action="store_true")
parser.add_argument('-c', '--comment', help='Optional comment to add to log', default='')
parser.add_argument('-l', '--logfile', help='Name of logfile to write to. Default=log.txt', default='log.txt')
parser.add_argument('-g', '--graphical', help='Plots the amplitude of n-modes against iteration number', action="store_true")
args = parser.parse_args()

if args.test:
	save_to_log = False
else:
	save_to_log = True
logfile = args.logfile
graphical = args.graphical
"""
====================
===INITIALISATION===
====================
"""

run_begin = datetime.now()
dt_string = run_begin.strftime("%d/%m/%Y %H:%M:%S")
#Begin timing
start_t = time.time()

#Allows for extra readability when calling amplitude values
current = 1
previous = 0

#Set up parameters
Nz = params.Nz # Vertical resolution (Number of z-levels)
Nn = params.Nn # Number of Fourier modes
Ra = params.Ra # Rayleigh number (measures convective driving)
Pr = params.Pr # Prandtl number (ratio of viscous to thermal diffusion)
a = params.a # Aspect ratio of the box
nsteps = params.nsteps # Total number of steps to run simulation for
output_z = math.floor(params.z_percent*Nz) # 
output_n = 1
dz = 1.0/(Nz-1) # Spacing between adjacent z-levels
oodz2 = 1./dz**2 #Constant for 1/dz^2 to avoid repeated recalculation
c = np.pi / a # constant to avoid repeated calculation

# Initialise arrays for the problem variables.
psi = np.zeros(shape=(Nn, Nz)) #Velocity Streamfunction, psi[n][z] = nth mode of the zth level
omega = np.zeros(shape=(Nn, Nz)) # Vorticity, omega[n][z] = nth mode of the zth level
temp = np.zeros(shape=(Nn, Nz)) # Temperature, temp[n][z] = nth mode of the zth level

#Initialise arrays for the time derivatives of temperature and vorticity.
# arrays called as: var_dt[step][n][z]
# where step is current or previous, n is mode and z is z-level 
omega_dt = np.zeros(shape=(2, Nn, Nz)) # time derivative of vorticity
temp_dt = np.zeros(shape=(2, Nn, Nz)) #time derivative of temperature 

# Arrays for holding the amplitude of problem variables. 
# Shape of 2 for previous and current timestep
t_amp = np.zeros(shape=(2, Nn))
omega_amp = np.zeros(shape=(2, Nn))
psi_amp = np.zeros(shape=(2, Nn))

# Array to hold the z-value for each z-level
z_vals = np.zeros(shape=(Nz), dtype=float) # array to hold the height
for i in range(1, Nz+1):
	z_vals[i-1]=(i-1)*dz

# Arrays to hold the output values for the analysis step
temp_check = np.zeros(shape=(Nn))
omega_check = np.zeros(shape=(Nn))
psi_check = np.zeros(shape=(Nn))
temp_amps = np.zeros(shape=(Nn))
omega_amps = np.zeros(shape=(Nn))
psi_amps = np.zeros(shape=(Nn))
"""
====================
=TRIDIAGONAL SET-UP=
====================
"""
# Matrix defined by Eq 2.21 in Glatzmaier
# Arrays to hold the values of the tridiagonal matrix. 
sub = []
dia = np.zeros(shape=(Nz))
sup = []

# Super- and sub-diagonal arrays do not change with timestep or mode, so can be
# populated now.

# Populate the subdiagional array. final element=0, all others = -(1/dz^2)
for i in range(1, Nz-1):
	sub.append(-oodz2)
sub.append(0)

# Populate the superdiagonal array. First element=0, all others = -(1/dz^2)
sup.append(0)
for i in range(1, Nz-1):
	sup.append(-oodz2)

#Calculate the timestep. Constraint 2.19 is dt < (dz^2)/4 to avoid numerical instability
if Pr <= 1: 
	dt = 0.9*((dz*dz)/4) #90% of Constaint 2.19 for safety.
else: #if Prandtl > 1, divide 2.19 by Pr to avoid numerical instability 
	dt = 0.9*((dz*dz)/(4*Pr))

#Add to log.txt the parameters of this simulation run.
if(save_to_log):
	with open(logfile, 'a') as log:
		log.write("====================\n\n")
		log.write("#"+dt_string+"\t"+args.comment+"\n")
		log.write("Nn = {}\t Nz = {}\t a = {}\n".format(Nn, Nz, a))
		log.write("Ra = {}\t Pr = {}\n".format(Ra, Pr))
		log.write("output z = {}\t output n = {}\n".format(output_z, output_n))
		log.write("dt = {:.3e}\titerations = {:.0e}\n".format(dt, nsteps))

"""
====================
===INITIAL VALUES===
====================
"""
# Populate temperature array for initial temperatures (t=0) 
for n in range(0, Nn):
	if n==0: #for n=0, T(z) = 1 - z for all z
		for z in range(0, Nz):
			temp[0][z] = 1-z_vals[z]
	else: #for n>0, T_(n, z) = sin(pi*z) at t=0
		for z in range(0, Nz):
			temp[n][z] = np.sin(np.pi*z_vals[z])
"""
====================
======SIM LOOP======
====================
"""
if output_n==0:
	print("The output when n=0 is 0 for temperature amplitude and undefined for Vorticity and Psi")
	print("Please input a value for output_n > 0")
	exit(1)
elif output_n > Nn:
	print("Please input a value for output_n that is <= Nn.")
	exit(2)

print("I\t| temperature \t| vorticity\t| streamfunction")
for iteration in range(0, int(nsteps+1)):
	#print("Step No. {}, time: {}".format(iteration, t))
	#Update derivatives of temperature and vorticity for the current timestep
	
	temp_dt, omega_dt = finite_diff(temp_dt, omega_dt, psi, Nn, Nz, c, oodz2, temp, Ra, Pr, omega)

	# Update temperature and vorticity using Adams-Bashford Time Integration
	temp, omega = adams_bashford(temp, omega, temp_dt, omega_dt, dt, Nn, Nz)
	# print(omega)
	# Update velocity streamfunction by creating tridiagonal matrix and solving
	psi = update_streamfunction(psi, sub, dia, sup, omega, Nn, Nz, c, oodz2)

	for n in range(0, Nn):
		for z in range(0, Nz):
			temp_dt[previous][n][z] = temp_dt[current][n][z]
			omega_dt[previous][n][z] = omega_dt[current][n][z]

	#ANALYSIS OUTPUT
	if iteration % 250 == 0:
		t_amp[current] = 0
		omega_amp[current] = 0
		psi_amp[current] = 0
		for n in range(Nn):
			t_amp[current][n] = temp[n][output_z]
			omega_amp[current][n] = omega[n][output_z]
			psi_amp[current][n] = psi[n][output_z]
		if iteration != 0:		
			for n in range(1, Nn):	
				temp_check[n] = np.log(np.abs(t_amp[current][n])) - np.log(np.abs(t_amp[previous][n]))
				omega_check[n] = np.log(np.abs(omega_amp[current][n])) - np.log(np.abs(omega_amp[previous][n]))
				psi_check[n] = np.log(np.abs(psi_amp[current][n])) - np.log(np.abs(psi_amp[previous][n]))
			temp_amps = np.vstack((temp_amps, temp_check))
			omega_amps = np.vstack((omega_amps, omega_check))
			psi_amps = np.vstack((psi_amps, psi_check))
			# print(temp_amps)
			print("{}\t| {:.6f}\t| {:.6f}\t| {:.6f}".format(iteration, temp_check[output_n], omega_check[output_n], psi_check[output_n]))

		t_amp[previous] = t_amp[current]
		omega_amp[previous] = omega_amp[current]
		psi_amp[previous] = psi_amp[current]

	if np.isnan(np.any(temp)):
		print("Temp is NaN. Exiting on iteration {}.".format(iteration))
		if save_to_log:
			print("Sim ended early due to NaN temp. Ended on iteration {}".format(iteration), file=open(logfile, 'a'))
		break

if(save_to_log):
	with open(logfile, 'a') as log:
		print("Temp {:.6}\t Omega {:.6}\t Psi {:.6}".format(temp_check, omega_check, psi_check), file=log)

end_t = time.time()
t_delta = end_t - start_t
print("Completed {} iterations in {:.2f} seconds.".format(iteration, t_delta)) # type: ignore

if (graphical):
	# fig = plt.figure(figsize=(6, 8))
	# temp_ax = fig.add_subplot(311)
	# omega_ax = fig.add_subplot(312)
	# psi_ax = fig.add_subplot(313)
	# temp_ax.set_title("Temperature Amplitudes")
	# omega_ax.set_title("Vorticity Amplitudes")
	# psi_ax.set_title("Streamfunction Amplitudes")

	xdata = np.arange(0, nsteps+250, 250)
	print(xdata)
	print(temp_amps)
	print(omega_amps)
	print(psi_amps)
	# print(temp_amps)
	# temp_ax.plot(xdata, temp_amps)
	# plt.show()
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_title('n=1 mode')
	ax1.plot(xdata, temp_amps[:,1], label='Temperature', color='r')
	ax1.plot(xdata, omega_amps[:,1], label='Vorticity', color='b')
	ax1.plot(xdata, psi_amps[:,1], label='Streamfunction', color='g', linestyle=':', linewidth=3)
	ax1.legend()
	plt.show()





exit(0)
