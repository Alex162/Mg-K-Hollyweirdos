#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:54:43 2018

@author: alexkemp
"""

import numpy as np
import os
import matplotlib.pyplot as plt

import lamost
import utils

from astropy.io import fits
from astropy.table import Table
from scipy import (interpolate, optimize as op)
import logging
from vectorizer import polynomial

plt.close("all")
star_index = 245603

mg1, mg2 = 5160, 5190





catalog =lamost.load_catalog()

catalogbig = lamost.load_catalog('Ho2017_Catalog.fits')
wavelengths = lamost.common_wavelengths
N, P = (len(catalog), wavelengths.size)

all_observed_flux = np.memmap(
    os.path.join(lamost.LAMOST_PATH, "observed_flux.memmap"),
    mode="r", dtype='float32', shape=(N, P))

all_observed_ivar = np.memmap(
    os.path.join(lamost.LAMOST_PATH, "observed_ivar.memmap"),
    mode="r", dtype='float32', shape=(N, P))

all_model_flux = np.memmap(
    os.path.join(lamost.LAMOST_PATH, "model_flux.memmap"),
    mode="r", dtype="float32", shape=(N, P))



observed_flux = all_observed_flux[star_index]
observed_ivar = all_observed_ivar[star_index].copy()
observed_ivar[wavelengths >= 8900] = 0.0
model_flux = all_model_flux[star_index]

catalog["id"] = [each.strip() for each in catalog["id"]]
catalogbig["LAMOST_ID"] = [each.strip() for each in catalogbig["LAMOST_ID"]]

catalogbig_index = np.where(catalogbig["LAMOST_ID"] == catalog["id"][star_index])[0][0]


Teff=catalogbig['Teff'][catalogbig_index]
logg=catalogbig['logg'][catalogbig_index]
MH=catalogbig['MH'][catalogbig_index]
CM=catalogbig['CM'][catalogbig_index]
NM=catalogbig['NM'][catalogbig_index]
AM=catalogbig['AM'][catalogbig_index]
Ak=catalogbig['Ak'][catalogbig_index]
Mg_Guess=AM


def predict_spectrum(xdata, *params):
    teff, logg, fe_h, c_fe, n_fe, a_fe, ak, mg_fe = params
    labels = np.array([teff, logg, fe_h, c_fe, n_fe, a_fe, ak])
    flux = np.dot(lamost.theta, lamost.vectorizer(labels).T).T
    modified_labels = np.array([teff, logg, fe_h, c_fe, n_fe, mg_fe, ak])
    mg_mask = (mg2 >= lamost.common_wavelengths) \
        * (lamost.common_wavelengths >=mg1)
    mg_flux = np.dot(lamost.theta, lamost.vectorizer(modified_labels).T)[mg_mask].flatten()
    flux[:, mg_mask] = mg_flux
    #raise a
    
#    flux[mg_mask] = np.dot(lamost.theta[mg_mask], lamost.vectorizermod(modified_labels))
    
    #return flux, modified_labels, mg_mask, labels

    flux = flux.flatten()
    return flux    
    
    
#flux, modified_labels, mg_mask,labels=predict_spectrum(Teff,logg,MH,CM,NM,AM,Ak,Mg_Guess)

import scipy.optimize as op

p0 = [Teff, logg, MH, CM, NM, AM, Ak, Mg_Guess]
#p0 = [AM - 1]
adjusted_ivar = observed_ivar/(1. + observed_ivar * lamost.scatters**2)
adjusted_sigma = np.sqrt(1.0/adjusted_ivar)
adjusted_sigma[~np.isfinite(adjusted_sigma)] = 10**6



def predict_spectrum2(xdata, *params):
    
    mg_fe, = params
    
    labels = np.array([Teff, logg, MH, CM, NM, AM, Ak])
    flux = np.dot(lamost.theta, lamost.vectorizer(labels).T).T
    modified_labels = np.array([Teff, logg, MH, CM, NM, mg_fe, Ak])
    mg_mask = (mg2 >= lamost.common_wavelengths) \
    * (lamost.common_wavelengths >=mg1)
    mg_flux = np.dot(lamost.theta, lamost.vectorizer(modified_labels).T)[mg_mask].flatten()
    flux[:, mg_mask] = mg_flux
    
    flux = flux.flatten()
    
    chisq = np.sum((flux - observed_flux)**2 * adjusted_ivar)/wavelengths.size
    
    print(params, chisq)
    return flux 


p_opt, p_cov = op.curve_fit(predict_spectrum, None, observed_flux, p0=p0, 
                            sigma=adjusted_sigma, absolute_sigma=True, method="lm",
                            factor=0.1, xtol=1e-10, ftol=1e-10)


new_model_flux = predict_spectrum(None, *p_opt)


fig = utils.plot_spectrum(wavelengths, observed_flux, observed_ivar, new_model_flux)
fig.axes[1].set_ylim(0.0, 1.2)
fig.suptitle('code_index: '+str(star_index) + ', LAMOST ID:'+str(catalog['id'][star_index]))



ax=fig.axes[1]
ax.plot(wavelengths, model_flux, c="b", drawstyle="steps-mid")

#flux2 = predict_spectrum2(None, *[-0.05])
#ax.plot(wavelengths, flux2, c="g", drawstyle="steps-mid")

plt.show()
