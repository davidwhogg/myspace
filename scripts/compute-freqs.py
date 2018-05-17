# Standard library
import logging
from os import path

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import h5py
from pyia import GaiaData

import gala.coordinates as gc
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic
import superfreq as sf

from logger import logger

# Global parameters
gc_frame = coord.Galactocentric(z_sun=0*u.pc, galcen_distance=8.3*u.kpc)
mw = gp.MilkyWayPotential()
H = gp.Hamiltonian(mw)
integrate_time = 2. * u.Gyr # HACK: hardcoded! ~128 orbital periods
cache_file = path.abspath('../cache/freqs.hdf5')

def cartesian_to_poincare_polar(w):
    r"""
    Convert an array of 6D Cartesian positions to Poincaré
    symplectic polar coordinates. These are similar to cylindrical
    coordinates.
    Parameters
    ----------
    w : array_like
        Input array of 6D Cartesian phase-space positions. Should have
        shape ``(norbits,6)``.
    Returns
    -------
    new_w : :class:`~numpy.ndarray`
        Points represented in 6D Poincaré polar coordinates.
    """

    R = np.sqrt(w[...,0]**2 + w[...,1]**2)
    # phi = np.arctan2(w[...,1], w[...,0])
    phi = np.arctan2(w[...,0], w[...,1])

    vR = (w[...,0]*w[...,0+3] + w[...,1]*w[...,1+3]) / R
    vPhi = w[...,0]*w[...,1+3] - w[...,1]*w[...,0+3]

    # pg. 437, Papaphillipou & Laskar (1996)
    sqrt_2THETA = np.sqrt(np.abs(2*vPhi))
    pp_phi = sqrt_2THETA * np.cos(phi)
    pp_phidot = sqrt_2THETA * np.sin(phi)

    z = w[...,2]
    zdot = w[...,2+3]

    new_w = np.vstack((R.T, pp_phi.T, z.T,
                       vR.T, pp_phidot.T, zdot.T)).T
    return new_w


def worker(task):
    i, w0 = task
    w0 = gd.PhaseSpacePosition.from_w(w0, galactic)

    logging.log(logging.DEBUG, 'Starting orbit {0}'.format(i))

    try:
        orbit = H.integrate_orbit(w0, dt=0.5*u.Myr, t1=0*u.Myr,
                                  t2=integrate_time)
    except Exception as e:
        logger.error('Failed to integrate orbit {0}\n{1}'
                     .format(i, str(e)))
        return i, np.full(3, np.nan)

    new_ws = cartesian_to_poincare_polar(orbit.w().T).T
    fs = [(new_ws[j] + 1j*new_ws[j+3]) for j in range(3)]

    freq = sf.SuperFreq(orbit.t.value, p=4)
    try:
        res = freq.find_fundamental_frequencies(fs)
    except Exception as e:
        logger.error('Failed to compute frequencies for orbit {0}\n{1}'
                     .format(i, str(e)))
        return i, np.full(3, np.nan)

    return i, res.fund_freqs


def callback(result):
    i, freqs = result

    with h5py.File(cache_file) as f:
        f['freqs'][i] = freqs


def main(pool):
    logger.debug('Starting...')

    # Load FGK stars with RVs
    # see FGK-select.ipynb
    g = GaiaData('../data/fgk.fits')
    c = g.skycoord
    galcen = c.transform_to(gc_frame)
    logger.debug('Data loaded...')

    w0 = gd.PhaseSpacePosition(galcen.cartesian)
    w0 = w0.w(galactic).T

    # Make sure output file exists
    if not path.exists(cache_file):
        with h5py.File(cache_file, 'w') as f:
            f.create_dataset('freqs', shape=(w0.shape[0], 3), dtype='f8')

    tasks = [(i, w0[i]) for i in range(w0.shape[0])]
    for r in pool.map(worker, tasks, callback=callback):
        pass


if __name__ == "__main__":
    import schwimmbad
    from argparse import ArgumentParser
    parser = ArgumentParser(description="")

    logger.setLevel(logging.DEBUG)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    main(pool)
