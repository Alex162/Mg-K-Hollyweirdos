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

C_K2012=np.array(pd.read_csv("C+K 2012.csv"))
Muc2012=np.array(pd.read_csv("Mucciarelli 2012.csv"))

ngc2808=np.array(pd.read_csv("NGC2808 Muc 2015.csv"))


LC=np.array(pd.read_csv("LasCampmgkabundances.csv"))
Teff=mgkarray[:,1]
logg=mgkarray[:,2]
metalicity=mgkarray[:,3]

mgkl=mgkarray[:,-2]
mgkb=mgkarray[:,-1]
globl=globarray[:,4]
globb=globarray[:,5]

C_KMg=C_K2012[:,1]
C_KMgu=C_K2012[:,2]
C_KK=C_K2012[:,3]
#C_KKu=C_K2012[:,4] (no uncertainties known)
MucMg=Muc2012[:,1]
MucMgu=Muc2012[:,2]
MucK=Muc2012[:,3]
MucKu=Muc2012[:,4]

ngc2808Mg=ngc2808[:,0]
ngc2808K=ngc2808[:,1]

LCMg=LC[:,1]
LCK=LC[:,2]


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


plt.figure(4)

plt.scatter(C_KMg, C_KK, label= " NGC 2419 Kirby & Cohen 2012", s=25, c='gray')
#plt.errorbar(C_KMg, C_KK, xerr=C_KMgu, linestyle="None")
(_, caps, _)=plt.errorbar(C_KMg, C_KK, xerr=C_KMgu, linestyle="None", linewidth=.8, capsize=4,zorder=4, c='gray')
for cap in caps:
    cap.set_markeredgewidth(.8)
    
plt.scatter(MucMg, MucK, label= "NGC 2419 Mucciarreli 2012", s=25, c='gray', marker='s')
#plt.errorbar(MucMg, MucK, xerr=MucMgu, yerr=MucKu, linestyle="None",linewidth=1)
(_, caps, _) = plt.errorbar(MucMg, MucK, xerr=MucMgu, yerr=MucKu, linestyle="None",linewidth=.8, capsize=4, zorder=3, c='gray')
for cap in caps:
    cap.set_markeredgewidth(.8)



plt.scatter(ngc2808Mg,ngc2808K, label="NGC 2808 Muc 2015", marker='o', facecolor='none', edgecolor='dimgray')
plt.scatter(LCMg, LCK, label= "Las Campanas 2018", c='k',marker='d',zorder=100)

plt.xlabel('[Mg/Fe]')
plt.ylabel('[K/Fe]')

x1=-1.2
y1=-0.4
x2=1
y2=1.8
m=(y2-y1)/(x2-x1)
c=y2-m*x2
lineeqn= "y=" + str(m) + "x+" + str(c)

plt.annotate('Mg depleted population',xy=(-1.5,0.7))
plt.annotate('Mg normal population',xy=(-0.8,-0.3))
plt.annotate(lineeqn, xy= (-1.6, 0.-.3))
plt.plot([x1,x2],[y1,y2],'--', zorder=0, c='r')
#l=plt.legend(loc=1)
#l.set_zorder(1)

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




star_index = 245603
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



