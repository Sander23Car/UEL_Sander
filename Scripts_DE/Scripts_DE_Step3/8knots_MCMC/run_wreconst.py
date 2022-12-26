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

import argparse
import numpy as np
import os.path

#
#  Initializing the library objects, this must be called before 
#  any other library function.
#
Ncm.cfg_init ()

my_parser = argparse.ArgumentParser(description='Run W-Reconstruction models')

my_parser.add_argument('-n', '--nknots',      metavar='N', type=int,   required=True,  help='Number of knots, must be N=1 (XCDM) or N>2 W-Spline.')
my_parser.add_argument('-z', '--zf',          metavar='z', type=float, required=True,  help='Last redshift knot.')
my_parser.add_argument('-w', '--nwalkers',    metavar='W', type=int,   required=False, help='Number of walkers to use in the MCMC.', default = 1200)
my_parser.add_argument('-K', '--Omegak',      metavar=("mu", "sigma"),  nargs=2, type=float,  help='A Gaussian Omega_k prior.')
my_parser.add_argument('-f', '--flat',        action='store_true',                     help='Whether to use a flat model.')
my_parser.add_argument('-C', '--use-cmb',     action='store_true',                     help='Whether to use CMB shift parameter.')
my_parser.add_argument('-O', '--over-smooth', metavar='ovs', type=float,               help='Define the over-smooth to be used by the sampler, negative value indicate automatic calibration')
my_parser.add_argument('-I', '--init-sample', metavar='init.fits', type=str,           help='Define the catalog to use as initial sampler')
my_parser.add_argument('-D', '--dinterp',     action='store_true',                     help='Whether to disable interpolation in APES.')
my_parser.add_argument('-Q', '--dparam',      action='store_true',                     help='Whether to use desceleration parameter spline (QSpline) model.')
my_parser.add_argument('-H', '--kernel',      choices=["Cauchy", "ST3", "Gauss"],  default="Cauchy", help='Kernel type to use.')

# Execute the parse_args() method
args = my_parser.parse_args()

if args.use_cmb:
    data_str = "data_all"
else:
    data_str = "data_local"

if args.Omegak is not None:
    data_str = f"{data_str}_Omegak_{args.Omegak[0]}_{args.Omegak[1]}"

zf = args.zf
nknots = args.nknots

#
#  Creating the serialization object
#
ser = Ncm.Serialize.new (0)

#
#  New homogeneous and isotropic cosmological model NcHICosmoDEXcdm 
#
if nknots == 1:
    cosmo = Nc.HICosmo.new_from_name (Nc.HICosmo, "NcHICosmoDEXcdm")
    cosmo.omega_x2omega_k ()

    cosmo.param_set_by_name ("H0",     70.0)
    cosmo.param_set_by_name ("Omegab",  0.05)
    cosmo.param_set_by_name ("Omegac",  0.25)
    cosmo.param_set_by_name ("Omegak",  0.00)

    cosmo.props.H0_fit     = True
    cosmo.props.Omegac_fit = True
    cosmo.props.Omegax_fit = not args.flat
    cosmo.props.w_fit      = True
    
    model_str = "xcdm"

elif nknots > 2:
    if not args.dparam:
        cosmo = Nc.HICosmo.new_from_name (Nc.HICosmo, "NcHICosmoDEWSpline{'w-length':<%d>, 'z1':<0.2>, 'zf':<%f>}" % (nknots, zf))
        cosmo.omega_x2omega_k ()
    
        cosmo.param_set_by_name ("H0",     70.0)
        cosmo.param_set_by_name ("Omegab",  0.05)
        cosmo.param_set_by_name ("Omegac",  0.25)
        cosmo.param_set_by_name ("Omegak",  0.00)

        cosmo.props.H0_fit     = True
        cosmo.props.Omegac_fit = True
        cosmo.props.Omegax_fit = not args.flat

        last_w_i = -1
        for i in range (nknots):
            _, w_i = cosmo.param_index_from_name ("w_%d" % i)
            cosmo.param_set_ftype (w_i, Ncm.ParamType.FREE)
            last_w_i = w_i
        cosmo.param_set (last_w_i, 0.0)    
    
        model_str = f"wspline{nknots}_zf{zf}"
    else:
        cosmo = Nc.HICosmo.new_from_name (Nc.HICosmo, "NcHICosmoQSpline{'qparam-length':<%d>, 'zf':<%f>}" % (nknots, zf))
    
        cosmo.param_set_by_name ("H0",     70.0)
        cosmo.param_set_by_name ("Omegat",  1.00)

        cosmo.props.H0_fit     = True
        cosmo.props.asdrag_fit = True
        cosmo.props.Omegat_fit = not args.flat

        last_q_i = -1
        for i in range (nknots):
            _, q_i = cosmo.param_index_from_name ("qparam_%d" % i)
            cosmo.param_set_ftype (q_i, Ncm.ParamType.FREE)
    
        model_str = f"qspline{nknots}_zf{zf}"            
else:
    print ("nknots cannot be == 2")
    exit (-1)

if args.flat:
    model_str = f"{model_str}_flat"

filename_base = f"wreconst_{model_str}_{data_str}"
filename_log = f"{filename_base}.log"

Ncm.cfg_set_logfile (filename_log)

#
#  Creating a new Distance object optimized to redshift 2.
#
dist = Nc.Distance (zf = zf)

#
# SNIa model
#
snia_id = Nc.DataSNIAId.COV_PANTHEON_PLUS_SH0ES_SYS_STAT
snia_model = Nc.SNIADistCov.new_by_id (dist, snia_id)

#
#  Creating a new Modelset and set cosmo as the HICosmo model to be used.
#
progress_file = f"{filename_base}_progress.mset"
mset = None
if os.path.exists (progress_file):
    mset = Ncm.MSet.load (progress_file, ser)
else:
    mset = Ncm.MSet ()
    mset.set (cosmo)
    mset.set (snia_model)
mset.prepare_fparam_map ()

#
#  Creating a new Dataset and add snia to it.
#
dset = Ncm.Dataset ()


#
# Adding SNIa data 
#
data_snia = Nc.DataSNIACov.new_from_cat_id (snia_id, False)
data_snia0 = data_snia.apply_filter_sh0es_z (0.01, True)
data_snia = data_snia0

dset.append_data (data_snia)

#
# Adding BAO data
# 

bao_samples = [Nc.DataBaoId.RDV_BEUTLER2011,
               Nc.DataBaoId.EMPIRICAL_FIT_ROSS2015,
               Nc.DataBaoId.DTR_DHR_SDSS_DR12_2016_DR16_COMPATIBLE,
               Nc.DataBaoId.DTR_DHR_SDSS_DR16_LRG_2021,
               Nc.DataBaoId.DTR_DHR_SDSS_DR16_QSO_2021,
               Nc.DataBaoId.EMPIRICAL_FIT_1D_SDSS_DR16_ELG_2021,
               Nc.DataBaoId.EMPIRICAL_FIT_2D_SDSS_DR16_LYAUTO_2021,
               Nc.DataBaoId.EMPIRICAL_FIT_2D_SDSS_DR16_LYXQSO_2021]

for bao_in in bao_samples:
    bao = Nc.data_bao_create (dist, bao_in)
    dset.append_data (bao)

#
# Adding H data
# 

for H_id in [Nc.DataHubbleId.GOMEZ_VALENT_COMP2018]:
    Hdata = Nc.DataHubble.new_from_id (H_id)
    dset.append_data (Hdata)

#
# Adding CMB distance priors
# 
if args.use_cmb:
    cmb_dp = Nc.DataCMBDistPriors.new_from_id (dist, Nc.DataCMBId.DIST_PRIORS_WMAP9) 
    dset.append_data (cmb_dp)

#
#  Creating a Likelihood from the Dataset.
#
lh = Ncm.Likelihood (dataset = dset)

if args.Omegak is not None:
    Omega_k0_func = Ncm.MSetFuncList.new_ns_name ("NcHICosmo", "E2Omega_k", None)
    Okprior = Ncm.PriorGaussFunc.new (Omega_k0_func, args.Omegak[0], args.Omegak[1], 0.0)
    lh.priors_add (Okprior)

if not args.dparam:
    Omega_x0_func = Ncm.MSetFuncList.new_ns_name ("NcHICosmoDE", "Omega_x0", None) 

if not args.flat and not args.dparam:
    Omega_x0_prior = Ncm.PriorFlatFunc.new (Omega_x0_func, 0.0, 1.0, 0.1, 1.0)
    lh.priors_add (Omega_x0_prior)

#
#  Logging knots
#
if nknots > 2 and not args.dparam:
    alpha_v = cosmo.get_alpha ()
    alpha_a = np.array (alpha_v.dup_array ())
    print ("# Knots-z: ", str(np.expm1 (alpha_a)))

#
# Additional functions of mset to be computed during the sampling 
# process.
#
mfunc_oa = Ncm.ObjArray.new()
if not args.dparam:
    mfunc_oa.add (Omega_x0_func)

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

fit.run_restart (Ncm.FitRunMsgs.SIMPLE, 1.0e-3, 0.0, mset_save, progress_file)

#
#  Printing fitting informations.
#
fit.log_info ()

#
#  Calculating the parameters covariance using numerical differentiation.
#
#fit.numdiff_m2lnL_covar ()

#
#  Printing the covariance matrix.
# 
#fit.log_covar ()

#
# Setting single thread calculation.
#
Ncm.func_eval_set_max_threads(3)
Ncm.func_eval_log_pool_stats()

#
# Walkers configuration
# 

nwalkers = args.nwalkers
walker = Ncm.FitESMCMCWalkerAPES.new (nwalkers, mset.fparams_len ())

if args.kernel == "Cauchy":
    walker.set_k_type (Ncm.FitESMCMCWalkerAPESKType.CAUCHY)
elif args.kernel == "ST3":
    walker.set_k_type (Ncm.FitESMCMCWalkerAPESKType.ST3)
else:
    walker.set_k_type (Ncm.FitESMCMCWalkerAPESKType.GAUSS)

#walker.set_cov_fixed_from_mset (mset)
#walker.set_cov_robust_diag ()
walker.set_cov_robust ()
#walker.set_local_frac (0.04)
if args.dinterp:
    walker.use_interp (False)

#
# Initial sampling
#

if args.init_sample is not None:
    kernel = Ncm.StatsDistKernelST.new (mset.fparams_len (), 3.0)
    sd = Ncm.StatsDistVKDE.new (kernel, Ncm.StatsDistCV.NONE)
    init_mcat = Ncm.MSetCatalog.new_from_file_ro (args.init_sample, 0)
    init_sampler = Ncm.MSetTransKernCat.new (init_mcat, sd)
    init_sampler.set_mset (mset)
    init_sampler.set_prior_from_mset ()
    init_sampler.set_sampling (Ncm.MSetTransKernCatSampling.CHOOSE)
else:
    init_sampler = Ncm.MSetTransKernGauss.new (0)
    init_sampler.set_mset (mset)
    init_sampler.set_prior_from_mset ()
    init_sampler.set_cov_from_rescale (1.0e-1)

fitscat = f"{filename_base}_esmcmc_{nwalkers}.fits"

walker.set_over_smooth (args.over_smooth)
print (f"# === Setting over smooth to {args.over_smooth}")

esmcmc  = Ncm.FitESMCMC.new_funcs_array (fit, nwalkers, init_sampler, walker, Ncm.FitRunMsgs.SIMPLE, mfunc_oa)

esmcmc.set_nthreads (4)
esmcmc.set_data_file (fitscat)

esmcmc.start_run ()
esmcmc.run_lre (100, 5.0e-3)
esmcmc.end_run ()

esmcmc.mean_covar ()
fit.log_covar ()
