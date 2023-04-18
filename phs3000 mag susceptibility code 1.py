# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:06:54 2023

@author: rache
"""

#Python workshop
#Rachel Woo, 6 March 2023

import numpy as np
import matplotlib.pyplot as plt
import monashspa.PHS3000 as spa


#Part A

'''
c = np.array([
    [1.1, 2.1, 3.1],
    [1.2, 2.2, 3.2],
    [1.3, 2.3, 3.3],
    [1.4, 2.4, 3.4],
    [1.5, 2.5, 3.5]
])

x = c[:,0]
y = c[:, 1]
u_y = c[:, 2]

print(x)
print(y)
print(u_y)

print(x < 1.3)

x_subset = x[x<1.3]

print(x_subset)

y_subset = y[x<1.3]
u_y_subset = u_y[x<1.3]

print(y_subset)
print(u_y_subset)

x_subset_2 = x[~((x>1.1)&(x<1.5))]
print(x_subset_2)

y_subset_2 = y[~((x>1.1)&(x<1.5))]
u_y_subset_2 = u_y[~((x>1.1)&(x<1.5))]

print(y_subset_2)
print(u_y_subset_2)
'''

#Part B
'''
data = spa.tutorials.fitting.part_b_data

x = data[:,0]
y = data[:, 1]
u_y = data[:, 2]

plt.figure(1)
plt.title("Figure 1: Non-linear plot of silver decay data")
plt.errorbar(x, y, yerr=u_y, marker="o", linestyle ='none', label = "activity data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(bbox_to_anchor=(1,1))


#Part B.1
#eq 5: A(t) = A_0*exp(-lambda*t)

linear_model = spa.make_lmfit_model('m*x+c')
linear_params = linear_model.make_params(m=1, c=2)

decay_model = spa.make_lmfit_model('A_0*np.exp(-l*x)')
decay_params = decay_model.make_params(A_0=21,l=0.2)

fit_results = spa.model_fit(decay_model, decay_params, x, y, u_y=u_y)

spa.get_fit_parameters(fit_results)
fit_results.best_fit
fit_results.eval_uncertainty(sigma=1)

plt.plot(x, fit_results.best_fit, marker="None", linestyle="-", color="black",label="nonlinear fit")
plt.fill_between(x,fit_results.best_fit-fit_results.eval_uncertainty(sigma=1),fit_results.best_fit+fit_results.eval_uncertainty(sigma=1), color="lightgrey",label="uncertainty in nonlinear fit")
plt.legend(bbox_to_anchor=(1,1))
'''
'''
#Part B.2
data = spa.tutorials.fitting.part_b_data
x = (data[:,0])
y = np.log(data[:, 1])
u_y = (data[:, 2])/data[:, 1]

#m is logm, c is log A_0

plt.figure(2)
plt.title("Figure 2: linear plot of silver decay data")
plt.errorbar(x, y, yerr=u_y, marker="o", linestyle ='none', label = "linear activity data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(bbox_to_anchor=(1,1))

linear_model = spa.make_lmfit_model('m*x+c')
linear_params = linear_model.make_params(m=-1, c=3)

fit_results = spa.model_fit(linear_model, linear_params, x_log, y_log, u_y=u_y_log)

spa.get_fit_parameters(fit_results)
fit_results.best_fit
fit_results.eval_uncertainty(sigma=1)

plt.plot(x_log, fit_results.best_fit, marker="None", linestyle="-", color="black",label="linear fit")
plt.fill_between(x,fit_results.best_fit-fit_results.eval_uncertainty(sigma=1),fit_results.best_fit+fit_results.eval_uncertainty(sigma=1), color="lightgrey",label="uncertainty in linear fit")

plt.legend(bbox_to_anchor=(1,1))
'''

#Part B.3
'''
data = spa.tutorials.fitting.part_b_data

x = data[:,0]
y = data[:, 1]
u_y = data[:, 2]


decay_model = spa.make_lmfit_model('A_0*np.exp(-l*x)')
decay_params = decay_model.make_params(A_0=21,l=0.2)

fit_results = spa.model_fit(decay_model, decay_params, x, y, u_y=u_y)

spa.get_fit_parameters(fit_results)

non_lin_y_fit = fit_results.best_fit
u_non_lin_y_fit = fit_results.eval_uncertainty(sigma=1)

y_residuals = non_lin_y_fit-y
y_0 = np.zeros(35)

plt.figure(3)
plt.title("Figure 3: Residuals of non-linear plot of silver decay data")
plt.errorbar(x, y_residuals, yerr=u_y, marker="o", linestyle ='none', label = "Residuals")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(bbox_to_anchor=(1,1))
plt.plot(x, y_0, marker="None", linestyle="-", color="black",label="y=0")
plt.legend(bbox_to_anchor=(1,1))
plt.show



x_log = (data[:,0])
y_log = np.log(data[:, 1])
u_y_log = (data[:, 2])/data[:, 1]

linear_model = spa.make_lmfit_model('m*x+c')
linear_params = linear_model.make_params(m=-1, c=3)

fit_results = spa.model_fit(linear_model, linear_params, x_log, y_log, u_y=u_y_log)

spa.get_fit_parameters(fit_results)
fit_results.best_fit
fit_results.eval_uncertainty(sigma=1)

lin_y_fit_log = fit_results.best_fit
u_lin_y_fit_log = fit_results.eval_uncertainty(sigma=1)

y_residuals_log = lin_y_fit_log-y_log
y_0_log = np.zeros(35)

plt.figure(4)
plt.title("Figure 4: Residuals of linear plot of silver decay data")
plt.errorbar(x_log, y_residuals_log, yerr=u_y_log, marker="o", linestyle ='none', label = "Residuals")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(bbox_to_anchor=(1,1))
plt.plot(x_log, y_0_log, marker="None", linestyle="-", color="black",label="y=0")

plt.legend(bbox_to_anchor=(1,1))
plt.show()

#Just from visual observation, we can't give a good comparison of which is a better plot as the y axis are scaled differently, one is linear and one is logarithmic. However, both models appear quite well fit and reasonable.

# For a nuclear decay experiment, it would be more important to have more accurate data early on in the experiment as the changes in activity are more drastic, therefore errors in the measurement would have a larger effect.
'''

#Part C.1
'''
data = spa.tutorials.fitting.part_c1_data

x = data[:,0]
y = data[:,1]

plt.figure(5)
plt.title("Figure 5: Plot of raw data")
plt.plot(x, y, marker="o", linestyle ='none', label = "activity data")
plt.xlabel("x")
plt.ylabel("y")

model1 = spa.make_lmfit_model('A*np.exp(-(x-B)**2/D**2)', name='Gaussian 1')
model2 = spa.make_lmfit_model('F*np.exp(-(x-G)**2/H**2)', name='Gaussian 2')
model3 = spa.make_lmfit_model('c+0*x', name='Offset')
model = model1 + model2 + model3
params = model.make_params(A=1,B=90,c=2.5,D=1,F=1,G=170,H=1)
params.add('H', expr='D')
fit_results = spa.model_fit(model, params, x, y)

spa.get_fit_parameters(fit_results)
fit_results.best_fit
fit_results.eval_uncertainty(sigma=1)

plt.plot(x, fit_results.best_fit, marker="None", linestyle="-", color="black",label="nonlinear fit")
plt.fill_between(x,fit_results.best_fit-fit_results.eval_uncertainty(sigma=1),fit_results.best_fit+fit_results.eval_uncertainty(sigma=1), color="lightgrey",label="uncertainty in nonlinear fit")
plt.legend(bbox_to_anchor=(1,1))

for component_name, component_y_values in fit_results.eval_components().items():
    plt.plot(x, component_y_values, label=component_name)
'''

#Part C.2
data = spa.tutorials.fitting.part_c2_data

t = data[:,0]
A = data[:,1]
u_A = data[:,2]

plt.figure(6)
plt.title("Figure 6: Non-linear plot of dual isotope silver decay data")
plt.plot(t, A, marker="o", linestyle ='none', label = "activity data")
plt.xlabel("t")
plt.ylabel("A")

ag110_model = spa.make_lmfit_model("A_0*np.exp(-x*np.log(2)/l)", prefix="AG110_", name = 'AG110')
ag108_model = spa.make_lmfit_model("A_0*np.exp(-x*np.log(2)/l)", prefix="AG108_", name = 'AG108')
offset = spa.make_lmfit_model('c+0*x', name='Offset')
model = ag110_model + ag108_model + offset
params = model.make_params(AG110_A_0=A[0],AG110_l=20,AG108_A_0=A[10],AG108_l=140)
params.add('c',value=1,min=0,max=10)
fit_results = spa.model_fit(model, params, x=t, y=A, u_y=u_A)

spa.get_fit_parameters(fit_results)
fit_results.best_fit
fit_results.eval_uncertainty(sigma=1)

plt.plot(t, fit_results.best_fit, marker="None", linestyle="-", color="black",label="nonlinear fit")
plt.fill_between(t,fit_results.best_fit-fit_results.eval_uncertainty(sigma=1),fit_results.best_fit+fit_results.eval_uncertainty(sigma=1), color="lightgrey",label="uncertainty in nonlinear fit")
plt.legend(bbox_to_anchor=(1,1))

for component_name, component_y_values in fit_results.eval_components().items():
    plt.plot(t, component_y_values, label=component_name)
    plt.legend(bbox_to_anchor=(1,1))

fit_parameters=spa.get_fit_parameters(fit_results)
print(fit_parameters)
hl_110=fit_parameters['AG110_l']
u_hl_110=fit_parameters['u_AG110_l']
hl_108=fit_parameters['AG108_l']
u_hl_108=fit_parameters['u_AG108_l']
print('AG110 hl = ', hl_110 , ', AG108 hl = ', hl_108)

#Residuals plot
non_lin_A_fit = fit_results.best_fit
u_non_lin_A_fit = fit_results.eval_uncertainty(sigma=1)

A_residuals= non_lin_A_fit-A
y_0 = np.zeros(45)

plt.figure(7)
plt.title("Figure 7: Residuals of non-linear plot of silver decay data")
plt.errorbar(t, A_residuals, yerr=u_A, marker="o", linestyle ='none', label = "Residuals, non-linear")
plt.xlabel("t")
plt.ylabel("A")
plt.legend(bbox_to_anchor=(1,1))
plt.plot(t, y_0, marker="None", linestyle="-", color="black",label="y=0")

plt.legend(bbox_to_anchor=(1,1))
plt.show()

#as this is a non-linear fit, an attempt at a linear fit was made

t_log = data[:,0]
A_log = np.log(data[:,1])
u_A_log = data[:,2]/data[:,1]

linear_model = spa.make_lmfit_model('m*x+c')
linear_params = linear_model.make_params(m=-1, c=3)

fit_results = spa.model_fit(linear_model, linear_params, t_log, A_log, u_y=u_A_log)

spa.get_fit_parameters(fit_results)
fit_results.best_fit
fit_results.eval_uncertainty(sigma=1)

lin_A_fit_log = fit_results.best_fit
u_lin_A_fit_log = fit_results.eval_uncertainty(sigma=1)

A_residuals_log = lin_A_fit_log-A_log
y_0_log = np.zeros(45)

plt.figure(8)
plt.title("Figure 8: Residuals of linear plot of silver decay data")
plt.errorbar(t_log, A_residuals_log, yerr=u_A_log, marker="o", linestyle ='none', label = "Residuals, linear")
plt.xlabel("t")
plt.ylabel("A")
plt.legend(bbox_to_anchor=(1,1))
plt.plot(t_log, y_0_log, marker="None", linestyle="-", color="black",label="y=0")

plt.legend(bbox_to_anchor=(1,1))
plt.show()

#The results appear quite skewed compared to the non-linear fit, likely because of the multiple models for the plot.

#The estimates for the halflife are slightly different from the expected values but are in the ballpark. The plots however, look identical to the expected plots.







