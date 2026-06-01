"""Compute Planck-mean opacity for a power-law size distribution with cutoff amax.

Usage:
  from scripts.planck_mean import planck_mean_vs_T
  T, kP = planck_mean_vs_T(amax=0.1)

Or run as a script:
  python -m scripts.planck_mean --amax 0.1 --plot
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c

import os
import sys

# make notebooks directory importable so we can reuse aux_functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks')))
import aux_functions as aux
import dsharp_opac as opacity


def planck_mean_vs_T(amax: float, T_array: np.ndarray | None = None, q: float = 3.5,
                     sigma_g: float = 200.0, r: float | None = None, M_star: float | None = None):
    """Compute Planck-mean opacity kappa_P(T) for a power-law a^(4-q) truncated at `amax`.

    Returns (T_array, kP_array). Uses the package datafiles `default_opacities_smooth.npz`
    and `icefree_opacities_smooth.npz` (wet/dry) following the logic in `notebooks/opac_widget.py`.
    """

    # load default opacities
    with np.load(opacity.get_datafile('default_opacities_smooth.npz')) as d:
        a_w = d['a']
        lam_w = d['lam']
        k_abs_w = d['k_abs']

    with np.load(opacity.get_datafile('icefree_opacities_smooth.npz')) as d:
        a_d = d['a']
        k_abs_d = d['k_abs']

    if not np.allclose(a_w, a_d):
        raise RuntimeError('size grids in opacity data do not match')

    a = a_w
    lam = lam_w
    nu = c.c.cgs.value / lam

    # default temperature grid
    if T_array is None:
        T_array = np.geomspace(5.0, 1500.0, 200)

    # size distribution: power-law n(a) ~ a^{-q}, mass weighting -> a^(4-q)
    power_law = a**(4.0 - q)
    power_law[a > amax] = 0.0
    power_law = power_law / power_law.sum()

    # size-averaged absorption opacities (per gram of dust)
    k_abs_p_w = (k_abs_w.T * power_law[None, :]).sum(1)
    k_abs_p_d = (k_abs_d.T * power_law[None, :]).sum(1)

    # thresholds for wet/dry as in opac_widget
    if r is None:
        r = c.au.cgs.value
    if M_star is None:
        M_star = c.M_sun.cgs.value

    T_sat = aux.t_sat_water(sigma_g, M_star, r)

    kP = np.empty_like(T_array)
    for i, T in enumerate(T_array):
        Bnu = aux.planck_B_nu(nu, T)
        B = np.trapz(Bnu, x=nu)
        if T < T_sat:
            kP[i] = np.trapz(Bnu * k_abs_p_w, x=nu) / B
        else:
            kP[i] = np.trapz(Bnu * k_abs_p_d, x=nu) / B

    return T_array, kP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amax', type=float, required=True, help='maximum particle size in cm')
    parser.add_argument('--nT', type=int, default=200)
    parser.add_argument('--Tmin', type=float, default=5.0)
    parser.add_argument('--Tmax', type=float, default=1500.0)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    T_array = np.geomspace(args.Tmin, args.Tmax, args.nT)
    T, kP = planck_mean_vs_T(args.amax, T_array=T_array)

    if args.plot:
        plt.figure()
        plt.loglog(T, kP)
        plt.xlabel('T [K]')
        plt.ylabel('$\\kappa_P$ [cm$^2$/g]')
        plt.title(f'Planck-mean opacity, amax={args.amax} cm')
        plt.grid(True, which='both', ls=':')
        plt.show()
    else:
        for t,v in zip(T, kP):
            print(f"{t:.6g} {v:.6g}")


if __name__ == '__main__':
    main()
