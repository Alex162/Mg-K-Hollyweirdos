#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:25:28 2018

@author: alexkemp
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import astropy as asp
"""
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid.inset_locator import inset_axes
"""
import lamost
import utils
import csv
plt.ioff()
from astropy.io import fits
from astropy.table import Table
plt.close('all')

mgkarray=np.array(pd.read_csv("Mg-K-candidatesFINALCUT_nonHemmision.csv"))
globarray=np.array(pd.read_csv("globular_clusters.csv"))

Teff=mgkarray[:,1]
logg=mgkarray[:,2]
metalicity=mgkarray[:,3]

mgkl=mgkarray[:,-2]
mgkb=mgkarray[:,-1]
globl=globarray[:,4]
globb=globarray[:,5]

catalog = lamost.load_catalog()
wavelengths = lamost.common_wavelengths
N, P = (len(catalog), wavelengths.size)

# Open the data arrays
all_observed_flux = np.memmap(
    os.path.join(lamost.LAMOST_PATH, "observed_flux.memmap"),
    mode="r", dtype='float32', shape=(N, P))

all_observed_ivar = np.memmap(
    os.path.join(lamost.LAMOST_PATH, "observed_ivar.memmap"),
    mode="r", dtype='float32', shape=(N, P))

all_model_flux = np.memmap(
    os.path.join(lamost.LAMOST_PATH, "model_flux.memmap"),
    mode="r", dtype="float32", shape=(N, P))


binwidth=.05
plt.figure(1)
plt.hist(metalicity, bins=np.arange(-1.5, 0.5, binwidth),edgecolor='black')
plt.xticks(np.arange(-1.5, 0.5, binwidth)[::4])
plt.xlabel("[Fe/H]")
plt.title('[Fe/H] for candidate stars')
plt.show()

plt.figure(2)
plt.scatter(Teff ,logg , c=metalicity, cmap='plasma',s=40)
cbar=plt.colorbar()
cbar.set_label('[Fe/H]')
plt.xlabel('Teff')
plt.ylabel('Log(g)')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

plt.figure(3)
plt.scatter(mgkl, mgkb, label='Mg-K Stars', s=10)
plt.scatter(globl, globb, label='Globular Clusters', facecolors='none', edgecolors='k')
plt.xticks(np.arange(0, 361, 45)[::1])
plt.xlabel('Galactic Longitude')
plt.ylabel('Galactic Lattitude')
plt.legend(loc='lower right')
plt.show()


def fill_between_steps(ax, x, y1, y2=0, h_align='mid', **kwargs):
    """
    Fill between for step plots in matplotlib.

    **kwargs will be passed to the matplotlib fill_between() function.
    """

    # If no Axes opject given, grab the current one:

    # First, duplicate the x values
    xx = x.repeat(2)[1:]
    # Now: the average x binwidth
    xstep = np.repeat((x[1:] - x[:-1]), 2)
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    # Now: add one step at end of row.
    xx = np.append(xx, xx.max() + xstep[-1])

    # Make it possible to chenge step alignment.
    if h_align == 'mid':
        xx -= xstep / 2.
    elif h_align == 'right':
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = y1.repeat(2)#[:-1]
    if type(y2) == np.ndarray:
        y2 = y2.repeat(2)#[:-1]

    # now to the plotting part:
    return ax.fill_between(xx, y1, y2=y2, **kwargs)


def plot_spectrummod(wavelengths, observed_flux, observed_ivar, model_flux):

    gs = plt.GridSpec(2, 1, height_ratios=[1, 4])
    fig = plt.figure(figsize=(13, 4))
    ax_residual = plt.subplot(gs[0])
    ax_spectrum = plt.subplot(gs[1], sharex=ax_residual)

    observed_flux_error = observed_ivar**-2
    ax_spectrum.plot(wavelengths, observed_flux, c="k", drawstyle="steps-mid", label='LAMOST Spectra')
    fill_between_steps(ax_spectrum, wavelengths, 
        observed_flux - observed_flux_error, observed_flux + observed_flux_error,
        facecolor="k", alpha=0.5)

    model_flux_error = lamost.scatters
    ax_spectrum.plot(wavelengths, model_flux, c="r", drawstyle="steps-mid",label='Model Spectra')
    fill_between_steps(ax_spectrum, wavelengths, model_flux - model_flux_error,
        model_flux + model_flux_error, facecolor="r", alpha=0.5)

    residual_flux = observed_flux - model_flux
    residual_flux_error = np.sqrt(model_flux_error**2 + observed_flux_error**2)

    ax_residual.plot(wavelengths, residual_flux, c="k", drawstyle="steps-mid")
    fill_between_steps(ax_residual, wavelengths,
        residual_flux - residual_flux_error, residual_flux + residual_flux_error,
        facecolor="k", alpha=0.5)

    for ax in (ax_spectrum, ax_residual):
        ax.set_xlim(wavelengths[0], wavelengths[-1])

    ax_spectrum.set_ylim(0, 1.2)

    value = np.mean(np.abs(np.percentile(residual_flux, [1, 99])))
    ax_residual.set_ylim(-value, +value)

    # Hide x-label ticks on the residual axes
    plt.setp(ax_residual.get_xticklabels(), visible=False)

    ax_spectrum.set_xlabel("Wavelengths (Angstroms)")
    ax_spectrum.set_ylabel("Normalized flux")
    
    return fig, ax_spectrum




star_index = 405638
observed_flux = all_observed_flux[star_index]
observed_ivar = all_observed_ivar[star_index]
model_flux = all_model_flux[star_index]


fig, ax = plot_spectrummod(wavelengths, observed_flux, observed_ivar, model_flux)
fig.axes[1].set_ylim(0.0, 1.2)
fig.suptitle('code_index: '+str(star_index) + ', LAMOST ID:'+str(catalog['id'][star_index]))
#plt.legend(bbox_to_anchor=(0.15, .5, 1.0, .102), loc=4)
plt.legend(loc='lower right')


#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#axins = zoomed_inset_axes(ax, 2.5, loc=1)

#Mg
axinsMg = inset_axes(ax, 3, 1 , loc=2, bbox_to_anchor=(0.22, 0.44), bbox_transform=ax.figure.transFigure) # no zoom
axinsMg.plot(wavelengths, observed_flux, 'k', wavelengths, model_flux, 'r')

x1, x2, y1, y2 = 5150, 5200, 0.6, 1.2 # specify the limits
axinsMg.set_xlim(x1, x2) # apply the x-limits
axinsMg.set_ylim(y1, y2) # apply the y-limits
plt.title('Mg')
#plt.yticks(visible=False)
#plt.xticks(visible=False)
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax, axinsMg, loc1=1, loc2=2, fc="none", ec="0.5")

#K

AxinsK = inset_axes(ax, 3, 1 , loc=2, bbox_to_anchor=(0.51, 0.44), bbox_transform=ax.figure.transFigure) # no zoom
AxinsK.plot(wavelengths, observed_flux, 'k', wavelengths, model_flux, 'r')

x1, x2, y1, y2 = 7650, 7720, 0.6, 1.2 # specify the limits
AxinsK.set_xlim(x1, x2) # apply the x-limits
AxinsK.set_ylim(y1, y2) # apply the y-limits
plt.title('K')
#plt.yticks(visible=False)
#plt.xticks(visible=False)
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax, AxinsK, loc1=1, loc2=2, fc="none", ec="0.5")



plt.draw()
plt.show()
raise a

# prepare the demo image
Z, extent = get_demo_image()
Z2 = np.zeros([150, 150], dtype="d")
ny, nx = Z.shape
Z2[30:30+ny, 30:30+nx] = Z

# extent = [-3, 4, -4, 3]
ax.imshow(Z2, extent=extent, interpolation="nearest",
          origin="lower")

axins = zoomed_inset_axes(ax, 6, loc=1) # zoom = 6
axins.imshow(Z2, extent=extent, interpolation="nearest",
             origin="lower")

# sub region of the original image
x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

plt.xticks(visible=False)
plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

