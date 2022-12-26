#!/usr/bin/env python

try:
  import gi
  gi.require_version('NumCosmo', '1.0')
  gi.require_version('NumCosmoMath', '1.0')
except:
  pass

from math import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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

#
# confidence_ellipse
#
def confidence_ellipse(mu, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


my_parser = argparse.ArgumentParser(description='Run W-Reconstruction models')

my_parser.add_argument('-O', '--over-smooth', metavar='ovs',      type=float, required=True,         help='Define the over-smooth to be used by the sampler, negative value indicate automatic calibration')
my_parser.add_argument('-K', '--kernel',      choices=["Cauchy", "ST3", "Gauss"],  default="Cauchy", help='Kernel type to use.')
my_parser.add_argument('-c', '--catalog', metavar='catalog.fits', type=str,   required=True,         help='Define the catalog to use in the calibration')

args = my_parser.parse_args()

mcat      = Ncm.MSetCatalog.new_from_file_ro (args.catalog, 0)
mcat_len  = mcat.len ()
nwalkers  = mcat.nchains ()
nadd_vals = mcat.nadd_vals ()
m2lnL_id  = mcat.get_m2lnp_var ()

assert mcat_len > nwalkers
assert args.over_smooth != 0.0

last_e = [mcat.peek_row (mcat_len - nwalkers + i) for i in range (nwalkers)]
ncols  = mcat.ncols ()
nvar   = ncols - nadd_vals
params = ["$" + mcat.col_symb (i) + "$" for i in range (nadd_vals, mcat.ncols ())]

if args.kernel == "Cauchy":
    kernel = Ncm.StatsDistKernelST.new (nvar, 1.0)
elif args.kernel == "ST3":
    kernel = Ncm.StatsDistKernelST.new (nvar, 3.0)
else:
    kernel = Ncm.StatsDistKernelGauss.new (nvar)

sd = Ncm.StatsDistVKDE.new (kernel, Ncm.StatsDistCV.NONE)

sd.set_over_smooth (fabs (args.over_smooth))
sd.set_cov_type (Ncm.StatsDistKDECovType.ROBUST)

if args.over_smooth < 0.0:
    sd.set_cv_type (Ncm.StatsDistCV.SPLIT)

#sd.set_const_kernel (mcat.peek_mset ().fparam_get_bound_matrix ())

m2lnL = []
for row in last_e:
    m2lnL.append (row.get (m2lnL_id))
    sd.add_obs (row.get_subvector (nadd_vals, nvar))

m2lnL_v = Ncm.Vector.new_array (m2lnL)
sd.prepare_interp (m2lnL_v)

ovs = sd.get_over_smooth ()
print (f"# === Setting over smooth to {ovs}")
        
rng = Ncm.RNG.new ()
xv = Ncm.Vector.new (nvar)
        
try_sample = []
for a in range (100):
    sd.sample (xv, rng)
    try_sample.append (xv.dup_array ())
            
try_sample = np.array (try_sample)

weights = np.array (sd.peek_weights ().dup_array ())
weights = weights / np.sum (weights)
max_w   = np.max (weights[np.nonzero(weights)])
min_w   = np.min (weights[np.nonzero(weights)])

#weights = 0.01 + 0.3 * (weights - min_w) / (max_w - min_w)
weights = weights / np.sum (weights)

for a in range (nvar):
    for b in range (a + 1, nvar):
        indices = np.array ([a, b])
        print (indices)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        nw = 0
        for i in range (0, int(sd.get_sample_size ())):
            y_i, cov_i, n_i, w_i = sd.get_Ki (i)
            mu  = np.array (y_i.dup_array ())
            cov = np.array ([[cov_i.get (i, j) for j in indices] for i in indices])
            cov = cov * 1.0
            w_i = weights[i]
            
            if w_i > 0.0:
                nw = nw + 1
                confidence_ellipse (mu[indices], cov, ax, edgecolor='red', facecolor='red', alpha = w_i)
        
        ax.scatter (try_sample[:,a], try_sample[:,b])
        print (f"# Number of RBF with non-zero weights: {nw}")
        plt.axis('auto')
        plt.xlabel (params[a])
        plt.ylabel (params[b])
        plt.grid ()
        plt.show ()
