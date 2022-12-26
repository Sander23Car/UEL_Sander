#!/usr/bin/env python

try:
  import gi
  gi.require_version('NumCosmo', '1.0')
  gi.require_version('NumCosmoMath', '1.0')
except:
  pass

import math
import argparse
import numpy as np

from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm

from tqdm import tqdm

from copy import copy
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

parser = argparse.ArgumentParser(description='Process mset catalogs')

parser.add_argument('-C', '--catalog', metavar='file.fits', 
                    help='catalog fits file', 
                    required = True)

parser.add_argument('-B', '--burnin', metavar='N', 
                    help='catalog burnin', type=int)

parser.add_argument('-T', '--thin', metavar='N', 
                    help='catalog thinning', type=int, default = 1)

parser.add_argument('-N', '--npoints', metavar='N', 
                    help='number of points in the z interval', type=int, default = 500)

parser.add_argument('--func',
                    help='which curve to plot', type=str,
                    required = True)

parser.add_argument('-M', '--mode', metavar="linear|log",
                    help='Set plot mode', default='log', type=str,
                    choices=["linear","log"])

parser.add_argument('--zi', metavar='z_i', 
                    help='Initial redshift, default 0.0', type=float, default = 0.0)

parser.add_argument('--zf', metavar='z_f', 
                    help='Final redshift, default 2.0', type=float, default = 2.0)

args = parser.parse_args()

Ncm.cfg_init ()

bin = 0
if args.burnin is not None:
    bin = args.burnin

cat = args.catalog
        
print (f"# Adding {cat} with burnin {bin}")

mcat = Ncm.MSetCatalog.new_from_file_ro (cat, bin)
nwalkers = mcat.nchains ()

if args.thin <= 0:
    args.thin = int (mcat.len () * args.npoints / 0.9e7)

print (f"#   Computing {args.func} for {int(mcat.len ()/args.thin)} rows, total length {mcat.len ()} thinning {args.thin}")

if not Ncm.MSetFuncList.has_ns_name ("NcHICosmo", args.func):
    raise ValueError (f"Function {args.func} not found")

func = Ncm.MSetFuncList.new_ns_name ("NcHICosmo", args.func)

mset = mcat.peek_mset ()
nadd_vals = mcat.nadd_vals ()

z_a = np.linspace (args.zi, args.zf, args.npoints)
z_v = Ncm.Vector.new_array (z_a)
f_v = Ncm.Vector.new (args.npoints)

z_total = []
f_total = []

if (mcat.len () * args.npoints / args.thin) > 1.0e7:
    print (f"Too many points: mcat.len * npoints / thin = "
           f"{mcat.len ()} * {args.npoints} / {args.thin} "
           f"== {mcat.len () * args.npoints / args.thin}, "
           f"increase thinning and/or burn-in.")
    exit(-1)    

stats = Ncm.StatsVec.new (args.npoints + 3, Ncm.StatsVecType.COV, False)

for i in tqdm(range (0, mcat.len (), args.thin)):
    row = mcat.peek_row (i)
    mset.fparams_set_vector_offset (row, nadd_vals)
    func.eval_vector (mset, z_v, f_v)
    
    mid = mset.get_id_by_ns ("NcHICosmo")
    cosmo = mset.peek (mid)
    
    stats.set (0, cosmo.H0())
    stats.set (1, cosmo.Omega_c0())
    stats.set (2, cosmo.Omega_k0())
    
    for i in range(args.npoints):
        stats.set (2 + i, f_v.get(i))
    stats.update_weight(1.0)
    #stats.append (f_v, False)
    z_total.append (z_v.dup_array ())
    f_total.append (f_v.dup_array ())

cov = stats.peek_cov_matrix (3)
cor = cov.cov_dup_cor ()

del cov
del mcat

z_total = np.array (z_total).flatten ()
f_total = np.array (f_total).flatten ()

fig, axes = plt.subplots(nrows=1, figsize=(12, 8), constrained_layout=True)

cmap = copy(plt.cm.plasma)
cmap.set_bad(cmap(0))
h, xedges, yedges = np.histogram2d(z_total, f_total, bins=[args.npoints, args.npoints], density=True)

vmax=1.0
vmin=1.0e-2

if args.mode == "log":

    pcm = axes.pcolormesh(xedges, yedges, h.T, cmap=cmap,
                             norm=LogNorm(vmax=vmax,vmin=vmin), rasterized=True)
    fig.colorbar(pcm, ax=axes, label="# points", pad=0)
    axes.set_xlabel("$z$")
    axes.set_ylabel(f"${func.peek_symbol ()}$")
    
elif args.mode == "linear":
    pcm = axes.pcolormesh(xedges, yedges, h.T, cmap=cmap,
                             vmax=vmax, vmin=vmin, rasterized=True)
    fig.colorbar(pcm, ax=axes, label="# points", pad=0)
    axes.set_xlabel("$z$")
    axes.set_ylabel(f"${func.peek_symbol ()}$")

plt.show()

#corr = np.array ([[cor.get(i, j) for i in range(args.npoints)] for j in range(args.npoints)])

#fig = plt.figure(figsize=(18,18), dpi = 200)
##sns.heatmap(cor_a, cmap=cmap)
##plt.matshow(cor_a)

#ax = sns.heatmap(corr,
 #   cmap=sns.diverging_palette(145, 300, as_cmap=True),
 #   vmin=-1.0, vmax=1.0,
 #   square=True, fmt='.2f',
 #   xticklabels=z_a)

#ticks = np.linspace(0.0, args.npoints, 20)
#ticklabels = ["%.2f" % (z) for z in np.linspace(args.zi, args.zf, 20)]
#ax.set_xticks(ticks, fontsize=6)
#ax.set_xticklabels(ticklabels, rotation=90, fontsize=8)
#ax.set_yticks(ticks, fontsize=6)
#ax.set_yticklabels(ticklabels, rotation=360, fontsize=8)
##ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
##ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
#plt.show()

#fig = plt.figure(figsize=(9,18), dpi = 200)

#corr_H0_f = [stats.get_cor(0, i + 3) for i in range(args.npoints)]

#plt.plot (z_a, corr_H0_f)
#plt.ylim((-1.0,1.0))
#plt.grid (visible=True, which='both', linestyle=':', color='0.75', linewidth=0.5)

#plt.show()

#fig = plt.figure(figsize=(9,18), dpi = 200)

#corr_Omega_c0_f = [stats.get_cor(1, i + 3) for i in range(args.npoints)]

#plt.plot (z_a, corr_Omega_c0_f)
#plt.ylim((-1.0,1.0))
#plt.grid (visible=True, which='both', linestyle=':', color='0.75', linewidth=0.5)

#plt.show()

#fig = plt.figure(figsize=(9,18), dpi = 200)

#corr_Omega_k0_f = [stats.get_cor(2, i + 3) for i in range(args.npoints)]

#plt.plot (z_a, corr_Omega_k0_f)
#plt.ylim((-1.0,1.0))
#plt.grid (visible=True, which='both', linestyle=':', color='0.75', linewidth=0.5)

#plt.show()


