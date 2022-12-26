#!/usr/bin/env python

try:
  import gi
  gi.require_version('NumCosmo', '1.0')
  gi.require_version('NumCosmoMath', '1.0')
except:
  pass

from math import *
import matplotlib.pyplot as plt
from gi.repository import GObject
from gi.repository import GLib
from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm

import numpy as np
from os.path import exists

#
#  Initializing the library objects, this must be called before 
#  any other library function.
#
Ncm.cfg_init ()

ser = Ncm.Serialize.new (0)

#
#  New homogeneous and isotropic cosmological model NcHICosmoDEXcdm 
#
nknots     = 10
zf         = 1080
fit_H0     = True
fit_Omegac = True
fit_Omegak = True

filebase   = "z_CMB_%.0f_%dknots_H0_%r_Omgc_%r_OmgK_%r" % (zf, nknots, fit_H0, fit_Omegac, fit_Omegak)
pfile      = filebase + ".mset"
logfile    = filebase + ".log"

Ncm.cfg_set_logfile (logfile)
cosmo = Nc.HICosmo.new_from_name (Nc.HICosmo, "NcHICosmoDEWSpline{'w-length':<%d>, 'z1':<0.2>, 'zf':<%f>}" % (nknots, zf))
cosmo.omega_x2omega_k ()

#
#  Setting values for the cosmological model, those not set stay in the
#  default values. Remeber to use the _orig_ version to set the original
#  parameters in case when a reparametrization is used.
#

#
# OO-like
#

cosmo.param_set_by_name ("H0",     70.0)
cosmo.param_set_by_name ("Omegab",  0.05)
cosmo.param_set_by_name ("Omegac",  0.25)
cosmo.param_set_by_name ("Omegak",  0.00)

#
#  Setting parameters Omega_c and w to be fitted.
#

cosmo.props.H0_fit     = fit_H0
cosmo.props.Omegac_fit = fit_Omegac
cosmo.props.Omegax_fit = fit_Omegak

for i in range (nknots):
    _, w_i = cosmo.param_index_from_name ("w_%d" % i)
    cosmo.param_set_ftype (w_i, Ncm.ParamType.FREE)


#
#  Creating a new Distance object optimized to redshift 2.
#
dist = Nc.Distance (zf = zf)

#
# SNIa model
#

snia = Nc.SNIADistCov.new (dist, 4)

snia.param_set_by_name ("alpha",   0.145)
snia.param_set_by_name ("beta",    3.16)
snia.param_set_by_name ("M1",    -19.0)
snia.param_set_by_name ("M2",    -19.2)

snia.param_set_by_name ("lnsigma_int_0", -2.52572864430826)
snia.param_set_by_name ("lnsigma_int_1", -2.22562405185792)
snia.param_set_by_name ("lnsigma_int_2", -2.00991547903123)
snia.param_set_by_name ("lnsigma_int_3", -2.30258509299405)

snia.param_set_by_name ("lnsigma_pecz",  -7.60021041)

for pname in ["alpha", "beta", "M1", "M2"]:

    F, i = snia.param_index_from_name (pname)
    assert F
    
    snia.param_set_ftype (i, Ncm.ParamType.FREE)
    

#
#  Creating a new Modelset and set cosmo as the HICosmo model to be used.
#
mset = Ncm.MSet ()

if exists (pfile):
    print ("# Loading progress...")
    mset = Ncm.MSet.load (pfile, ser)
    cosmo = mset.peek (mset.get_id_by_ns ("NcHICosmo"))
    snia  = mset.peek (mset.get_id_by_ns ("NcSNIADistCov"))
    assert cosmo
    assert snia
else:
    mset.set (cosmo)
    mset.set (snia)

mset.pretty_log ()

#
#  Creating a new Dataset and add snia to it.
#
dset = Ncm.Dataset ()


#
# Adding SNIa data 
#
data_snia = Nc.DataSNIACov.new (False)
data_snia.load_cat (Nc.DataSNIAId.COV_JLA_SNLS3_SDSS_SYS_STAT_CMPL)
dset.append_data (data_snia)

#
# Adding BAO data
# 

for bao_in in [Nc.DataBaoId.RDV_BEUTLER2011, Nc.DataBaoId.RDV_PADMANABHAN2012, Nc.DataBaoId.RDV_ANDERSON2012, Nc.DataBaoId.RDV_BLAKE2012]:
    bao = Nc.data_bao_create (dist, bao_in)
    dset.append_data (bao)

#
# Adding H data
# 

for H_id in [Nc.DataHubbleId.RIESS2008_HST, Nc.DataHubbleId.STERN2009, Nc.DataHubbleId.MORESCO2012_BC03]:
    Hdata = Nc.DataHubble.new_from_id (H_id)
    dset.append_data (Hdata)

#
# Adding CMB distance priors
# 

cmb_dp = Nc.DataCMBDistPriors.new_from_id (dist, Nc.DataCMBId.DIST_PRIORS_WMAP9) 
dset.append_data (cmb_dp)

#
#  Creating a Likelihood from the Dataset.
#
lh = Ncm.Likelihood (dataset = dset)

#
#  Logging knots
#
alpha_v = cosmo.get_alpha ()
alpha_a = np.array (alpha_v.dup_array ())
print (np.expm1 (alpha_a))

#
#  Creating a Fit object of type NLOPT using the fitting algorithm ln-neldermead to
#  fit the Modelset mset using the Likelihood lh and using a numerical differentiation
#  algorithm (NUMDIFF_FORWARD) to obtain the gradient (if needed).
#
fit = Ncm.Fit.new (Ncm.FitType.NLOPT, "ln-neldermead", lh, mset, Ncm.FitGradType.NUMDIFF_FORWARD)

#
#  Running the fitter printing messages.
#
mset_save = mset.dup (ser)

fit.run_restart (Ncm.FitRunMsgs.SIMPLE, 1.0e-3, 0.0, mset_save, pfile)

#
#  Printing fitting informations.
#
fit.log_info ()

#
#  Calculating the parameters covariance using numerical differentiation.
#
fit.numdiff_m2lnL_covar ()

#
#  Printing the covariance matrix.
# 
fit.log_covar ()
