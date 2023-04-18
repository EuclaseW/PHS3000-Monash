# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:28:58 2023

@author: rache
"""

#Python workshop
#Rachel Woo, 18 April 2023

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
import monashspa.PHS3000 as spa

cwd = os.getcwd()
filename = "mag_suscept_calibration.csv"
calibration = np.array(pandas.read_csv(cwd + "\\" + filename))
calibration_x = calibration[:,0]
calibration_u_x = calibration[:,1]
calibration_y = calibration[:,2]*10**-3
calibration_u_y = calibration[:,3]*10**-3

plt.figure(1)
plt.title("Calibration of relationship between current and magnetic field")
plt.errorbar(calibration_x, calibration_y, xerr=calibration_u_x, yerr=calibration_u_y, marker="o", markersize = "2", linestyle ='none', label = "Current and Magnetic Field Data")
plt.xlabel("Current (A)")
plt.ylabel("Magnetic Field (T)")
plt.legend(bbox_to_anchor=(1,1)) 

calibration_linear_model = spa.make_lmfit_model('m*x+c')
calibration_linear_params = calibration_linear_model.make_params(m=-1, c=3)

calibration_fit_results = spa.model_fit(calibration_linear_model, calibration_linear_params, calibration_x, calibration_y, u_y=calibration_u_y)

spa.get_fit_parameters(calibration_fit_results)
calibration_fit_results.best_fit
calibration_fit_results.eval_uncertainty(sigma=1)

plt.figure(2)
plt.title("Calibration of relationship with linear fit")
plt.errorbar(calibration_x, calibration_y, xerr=calibration_u_x, yerr=calibration_u_y, marker="o", markersize = "2", linestyle ='none', label = "Current and Magnetic Field Data")
plt.fill_between(calibration_x,calibration_fit_results.best_fit-calibration_fit_results.eval_uncertainty(sigma=1),calibration_fit_results.best_fit+calibration_fit_results.eval_uncertainty(sigma=1), color="lightgrey",label="uncertainty in linear fit")
plt.xlabel("Current (A)")
plt.ylabel("Magnetic Field (T)")
plt.legend(bbox_to_anchor=(1,1))

calibration_fit_parameters=spa.get_fit_parameters(calibration_fit_results)
print(calibration_fit_parameters)


cwd = os.getcwd()
filename = "mag_suscept_glass.csv"
glass = np.array(pandas.read_csv(cwd + "\\" + filename))
print(glass)
glass_x = glass[:,0]
glass_u_x = glass[:,1]
glass_y = glass[:,2]
glass_u_y = glass[:,3]

plt.figure(3)
plt.title("Change of measured mass of the glass testing tube")
plt.errorbar(glass_x, glass_y, xerr=glass_u_x, yerr=glass_u_y, marker="o", markersize = "2", linestyle ='none',capsize=2, label = "Mass and magnetic field data for glass testing tube")
plt.xlabel("Magnetic Field (T)")
plt.ylabel("Mass (kg)")
plt.legend(bbox_to_anchor=(1,1)) 









