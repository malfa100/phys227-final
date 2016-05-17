"""
File: final.py
Copyright (c) 2016 Andrew Malfavon
License: MIT
Description: Use 4th-order Runge-Kutta to solve coupled first order ODEs.
Various 2 and 3 dimensional graphs are created.
Local maxima for different c parameters are found and graphed with respect to c, creating a bifurcation diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D#used for 3D plot

class Rossler():
    def __init__(self, c, dt=0.001, T0=250, T=500):
        #initiates the given values from the problem
        self.c = c
        self.dt = dt
        self.T0 = T0
        self.T = T
        self.t = np.linspace(0, T, (int(T / float(dt)) + 1))
        self.x = np.zeros(len(self.t))
        self.y = np.zeros(len(self.t))
        self.z = np.zeros(len(self.t))
        self.a = self.b = 0.2

    def x_dot(self, t, x, y, z):
        #defines the given ODE for the derivative of x
        return -y - z

    def y_dot(self, t, x, y, z):
        #defines the given ODE for the derivative of y
        return x + (self.a * y)

    def z_dot(self, t, x, y, z):
        #defines the given ODE for the derivative of z
        return self.b + z * (x - self.c)

    def run(self):
        #fourth-order Runge-Kutta integration of the ODEs
        self.x[0] = self.y[0] = self.z[0] = 0#initial conditions

        for i in range(1, len(self.t)):#the one skips over the zero elements which have been set in the initial conditions
            #first-order Runge-Kutta
            k1_x = self.dt * self.x_dot(self.t[i - 1], self.x[i - 1], self.y[i - 1], self.z[i - 1])
            k1_y = self.dt * self.y_dot(self.t[i - 1], self.x[i - 1], self.y[i - 1], self.z[i - 1])
            k1_z = self.dt * self.z_dot(self.t[i - 1], self.x[i - 1], self.y[i - 1], self.z[i - 1])

            #second-order Runge-Kutta
            k2_x = self.dt * self.x_dot(self.t[i - 1] + (0.5 * self.dt), self.x[i - 1] + (0.5 * k1_x), self.y[i - 1] + (0.5 * k1_y), self.z[i - 1] + (0.5 * k1_z))
            k2_y = self.dt * self.y_dot(self.t[i - 1] + (0.5 * self.dt), self.x[i - 1] + (0.5 * k1_x), self.y[i - 1] + (0.5 * k1_y), self.z[i - 1] + (0.5 * k1_z))
            k2_z = self.dt * self.z_dot(self.t[i - 1] + (0.5 * self.dt), self.x[i - 1] + (0.5 * k1_x), self.y[i - 1] + (0.5 * k1_y), self.z[i - 1] + (0.5 * k1_z))

            #third-order Runge-Kutta
            k3_x = self.dt * self.x_dot(self.t[i - 1] + (0.5 * self.dt), self.x[i - 1] + (0.5 * k2_x), self.y[i - 1] + (0.5 * k2_y), self.z[i - 1] + (0.5 * k2_z))
            k3_y = self.dt * self.y_dot(self.t[i - 1] + (0.5 * self.dt), self.x[i - 1] + (0.5 * k2_x), self.y[i - 1] + (0.5 * k2_y), self.z[i - 1] + (0.5 * k2_z))
            k3_z = self.dt * self.z_dot(self.t[i - 1] + (0.5 * self.dt), self.x[i - 1] + (0.5 * k2_x), self.y[i - 1] + (0.5 * k2_y), self.z[i - 1] + (0.5 * k2_z))

            #fourth-order Runge-Kutta
            k4_x = self.dt * self.x_dot(self.t[i - 1] + self.dt, self.x[i - 1] + k3_x, self.y[i - 1] + k3_y, self.z[i - 1] + k3_z)
            k4_y = self.dt * self.y_dot(self.t[i - 1] + self.dt, self.x[i - 1] + k3_x, self.y[i - 1] + k3_y, self.z[i - 1] + k3_z)
            k4_z = self.dt * self.z_dot(self.t[i - 1] + self.dt, self.x[i - 1] + k3_x, self.y[i - 1] + k3_y, self.z[i - 1] + k3_z)

            #solutions from Runge-Kutta put into three arrays representing the three ODEs
            self.x[i] = self.x[i - 1] + (k1_x + (2 * k2_x) + (2 * k2_x) + k4_x) / 6
            self.y[i] = self.y[i - 1] + (k1_y + (2 * k2_y) + (2 * k2_y) + k4_y) / 6
            self.z[i] = self.z[i - 1] + (k1_z + (2 * k2_z) + (2 * k2_z) + k4_z) / 6
        return self.x, self.y, self.z


    def plotx(self):
        #plot x vs. t
        plt.plot(self.t, self.x)
        plt.title('x vs. t')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.axis([0, self.T, -12, 12])#sets axes

    def ploty(self):
        #plot y vs. t
        plt.plot(self.t, self.y)
        plt.title('y vs. t')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.axis([0, self.T, -12, 12])

    def plotz(self):
        #plot z vs. t
        plt.plot(self.t, self.z)
        plt.title('z vs. t')
        plt.xlabel('t')
        plt.ylabel('z')
        plt.axis([0, self.T, -12, 12])

    def plotxy(self):
        #plot y vs. x
        #t is from T0 to T
        plt.plot(self.x[int(self.T0 / float(self.dt)):len(self.t)], self.y[int(self.T0 / float(self.dt)):len(self.t)])
        #this and the next three graphs use t = [T0, T]
        plt.title('y vs. x')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis([-12, 12, -12, 12])

    def plotyz(self):
        #plot z vs. y
        #t is from T0 to T
        plt.plot(self.y[int(self.T0 / float(self.dt)):len(self.t)], self.z[int(self.T0 / float(self.dt)):len(self.t)])
        plt.title('z vs. y')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.axis([-12, 12, 0, 25])

    def plotxz(self):
        #plot z vs. x
        #t is from T0 to T
        plt.plot(self.x[int(self.T0 / float(self.dt)):len(self.t)], self.z[int(self.T0 / float(self.dt)):len(self.t)])
        plt.title('z vs. x')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.axis([-12, 12, 0, 25])

    def plotxyz(self):
        #3D x, y, z plot
        #t is from T0 to T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x[int(self.T0 / float(self.dt)):len(self.t)], self.y[int(self.T0 / float(self.dt)):len(self.t)], self.z[int(self.T0 / float(self.dt)):len(self.t)])
        ax.set_title('3D plot wit x, y, z axes')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim3d(-12, 12)
        ax.set_ylim3d(-12, 12)
        ax.set_zlim3d(0, 25)

    def findmaxima(self, x):
        #find a local maxima for a particular solution x(t)
        maxima = [max(x[int(self.T0 / float(self.dt)) + 1:len(self.t)])]
        #maxima is a one elment array of the max value of x(t) for t>T0
        for i in range(int(self.T0 / float(self.dt)) + 1, len(self.t) - 1):
            #adds points to the maxima array that are within + or - 10^-8 of this max value
            if ((x[i] - 1E-8 > x[i - 1]) and (x[i] - 1E-8 > x[i + 1])):
                maxima.append(x[i])
        return maxima

def plotmaxima(x_y_z):
    """plots the particular solution. x_y_z denotes if you use the solution x(t), y(t), or z(t).
    They are denoted by 0, 1, or 2 respectively."""
    plt.figure(figsize=(15, 15))#bigger plot
    c_range = np.linspace(2, 6, int(4 / 0.001))
    for c in c_range:
        #solves the ODE for each c in c_range
        values = Rossler(c, dt = 0.1)#dt=0.1 instead of 0.001 to prevent a ridiculous number of iterations
        solve = values.run()[x_y_z]
        max_values = values.findmaxima(solve)
        plt.plot(([c] * len(max_values)), max_values, 'ro', markersize=0.5)
        #max values vs. corresponding c values
    plt.axis([2, 6, 3, 12])
    plt.title('Asymptotic local maxima vs. c')
    plt.xlabel('c')
    plt.ylabel('Maxima')

def test_initial_cond():
    """x(0) = y(0) = z(0) = 0 leads to x_dot(0) = y_dot(0) = 0, z_dot(0) = b = 0.2"""
    sol = Rossler(2)
    A = sol.run()[0][0]
    B = sol.run()[1][0]
    C = sol.run()[2][0]
    assert (-B - C) == 0
    assert (A + 0.2 * B) == 0
    assert (0.2 + C * B - 2* C) == 0.2

def test_run():
    sol = Rossler(2)
    #Runge-Kutta solutions for t=500:
    A = sol.run()[0][-1]
    B = sol.run()[1][-1]
    C = sol.run()[2][-1]
    #numerical solutions for t=500 using Mathematica (shown in the notebook):
    mathematica_A = 3.85425
    mathematica_B = -0.360879
    mathematica_C = 1.1211
    assert abs(A - mathematica_A) < 0.001
    assert abs(B - mathematica_B) < 0.001
    assert abs(C - mathematica_C) < 0.001