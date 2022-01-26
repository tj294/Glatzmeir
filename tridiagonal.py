"""
Name: tridiagonal.py
Author: Tom Joshi-Cale

A function for solving a tridiagonal matrix, adapted from the FORTRAN subroutine
presented in Glatzmaier

"""

import numpy as np

def tridiagonal(sub, dia, sup, rhs):
	'''Solves a tridiagonal matrix equation of rank Nz.

	Inputs:
		sub - An array of length Nz containing the subdiagonal elements of the tridiagonal matrix
		
		dia - An array of length Nz containing the diagonal elements of the tridiagonal matrix
		
		sup - An array of length Nz containing the superdiagonal elements of the tridiagonal matrix
		
		rhs - An array of length Nz containing the RHS of the matrix equation
	Outputs:
		sol - The solution vector of length Nz
'''

	Nz = len(dia)
	wk1 = np.zeros(Nz)
	wk2 = np.zeros(Nz)
	sol = np.zeros(Nz)
	sub = np.append(1, sub)
	sup = np.append(sup, 1)

	wk1[0] = 1.0/dia[0]
	wk2[0] = sup[0] * wk1[0]

	for i in range(1, Nz-1):
		wk1[i] = 1/(dia[i] - sub[i]*wk2[i-1])
		wk2[i] = sup[i]*wk1[i]
	wk1[Nz-1] = 1/(dia[Nz-1] - sub[Nz-1] * wk2[Nz-2])

	sol[0] = rhs[0]*wk1[0]
	for i in range(1, Nz):
		sol[i] = (rhs[i] - sub[i]*sol[i-1])*wk1[i] 
	for i in range(Nz-2, -1, -1):
		sol[i] = sol[i] - wk2[i] * sol[i+1]
	return np.array(sol)


"""DEBUGGING STEPS FOR tridiagonal()
low = 1
high = 10
shape = (100)
sub = np.random.randint(low, high, shape-1)
dia = np.random.randint(10*low, 10*high, shape)
sup = np.random.randint(low, high, shape-1)
rand_sol = np.random.randint(low, high, shape)
print(rand_sol)
A = np.diag(sub, -1) + np.diag(dia, 0) + np.diag(sup, 1)
rhs = np.inner(A, rand_sol)
calc_sol = tridiagonal(sub, dia, sup, rhs)
print(calc_sol)
if np.all(np.isclose(rand_sol, np.array(calc_sol), rtol=0.001)):
	print("Success")
#"""

