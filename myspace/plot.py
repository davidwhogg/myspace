import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

from gala.mpl_style import hesperia, laguna, turbo_r  # noqa


class WedgeAnimation:

    def __init__(self, x_bins, v_bins,
                 wedge_size=20*u.deg, wedge_step=5*u.deg,
                 x_unit=u.pc, v_unit=u.km/u.s):

        # For internal representations of quantities:
        self.x_unit = x_unit
        self.v_unit = v_unit

        self._x_bins = u.Quantity(x_bins).to_value(x_unit)
        self._v_bins = u.Quantity(v_bins).to_value(v_unit)

        # Set up info to make wedge masks
        wedgie_edgies = np.arange(0, 360,
                                  wedge_step.to_value(u.deg))
        wedge_l = wedgie_edgies
        wedge_r = wedge_l + wedge_size.to_value(u.deg)
        self._wedge_iter = list(zip(wedge_l, wedge_r))

    def _animate_helper(self, X, V, axes=None, R_lim=None):
        R = np.sqrt(X[0]**2 + X[1]**2)
        phi = coord.Angle(np.arctan2(X[1], X[0]) * u.radian)
        phi = phi.wrap_at(360*u.deg).degree

        if R_lim is not None:
            R_mask = ((R > R_lim[0].to_value(self.x_unit)) &
                      (R <= R_lim[1].to_value(self.x_unit)))
        else:
            R_mask = None

        if axes is None:
            # set up figure and axes:
            fig, axes = plt.subplots(2, 2, figsize=(10, 9.1),
                                     constrained_layout=True)

        # setup background "image" in xy panel
        H, xe, ye = np.histogram2d(X[0], X[1], bins=self._x_bins)
        axes[0, 0].pcolormesh(xe, ye, H.T, cmap='Greys',
                              norm=mpl.colors.LogNorm())

        # setup overlay image in xypanel
        wedge_mesh = axes[0, 0].pcolormesh(
            xe, ye, H.T, cmap='hesperia_r',
            norm=mpl.colors.LogNorm())
        wedge_mesh.set_array(np.full_like(H.T, np.nan).ravel())

        # setup velocity panels
        vel_mesh = []
        for ax, VV in zip(axes.flat[1:], V):
            H, xe, ye = np.histogram2d(VV[0], VV[1], bins=self._v_bins)
            mesh = ax.pcolormesh(xe, ye, H.T, cmap='laguna_r',
                                 norm=mpl.colors.LogNorm())
            vel_mesh.append(mesh)
        self._vel_norm = None

        def animate(i):
            phi_l, phi_r = self._wedge_iter[i]

            if phi_r <= 360:
                mask = (phi > phi_l) & (phi <= phi_r)
            else:
                mask = (phi > phi_l) | ((phi+360) <= phi_r)

            if R_mask is not None:
                mask &= R_mask

            H, xe, ye = np.histogram2d(X[0, mask], X[1, mask],
                                       bins=self._x_bins)
            wedge_mesh.set_array(H.T.ravel())

            # velocity panels:
            for VV, mesh in zip(V, vel_mesh):
                H, xe, ye = np.histogram2d(VV[0][mask],
                                           VV[1][mask],
                                           bins=self._v_bins)
                mesh.set_array(H.T.ravel())

            if self._vel_norm is None:
                self._vel_norm = mpl.colors.LogNorm(0.8, np.nanpercentile(H, 99.9))
                for mesh in vel_mesh:
                    mesh.set_norm(self._vel_norm)

            return [wedge_mesh] + vel_mesh

        def init_func():
            return animate(0)

        return fig, axes, animate, init_func

    def make_vxyz(self, gal, R_lim=None, axes=None):
        # heliocentric Galactic position/velocity:
        X = gal.cartesian.xyz.to_value(self.x_unit)
        V = gal.velocity.d_xyz.to_value(self.v_unit)

        vel_idx = [(0, 1), (0, 2), (1, 2)]
        Vs = [(V[idx[0]], V[idx[1]]) for idx in vel_idx]

        fig, axes, animate_func, init_func = self._animate_helper(
            X, Vs, R_lim=R_lim, axes=axes)

        # Axis labels:
        axes[0, 0].set_xlabel(r'$x_\odot$ ' + f'[{self.x_unit:latex_inline}]')
        axes[0, 0].set_ylabel(r'$y_\odot$ ' + f'[{self.x_unit:latex_inline}]')

        names = 'xyz'
        for ax, idx in zip(axes.flat[1:], vel_idx):
            ax.set_xlabel('$v_{' + names[idx[0]] + r', \odot}$ ' + f'[{self.v_unit:latex_inline}]')
            ax.set_ylabel('$v_{' + names[idx[1]] + r', \odot}$ ' + f'[{self.v_unit:latex_inline}]')

        anim = FuncAnimation(
            fig, animate_func, frames=len(self._wedge_iter),
            interval=20, blit=True, repeat=True, init_func=init_func)

        return anim

    def make_myspace(self, gal, myspace1, myspace2, R_lim=None, axes=None):
        # heliocentric Galactic position/velocity:
        X = gal.cartesian.xyz.to_value(self.x_unit)
        V = gal.velocity.d_xyz.to_value(self.v_unit)

        pred_V1 = myspace1.get_model_v(V.T, X.T).T
        pred_V2 = myspace2.get_model_v(V.T, X.T).T

        Vs = [(V[0], V[1]),
              (pred_V1[0], pred_V1[1]),
              (pred_V2[0], pred_V2[1])]

        fig, axes, animate_func, init_func = self._animate_helper(
            X, Vs, R_lim=R_lim, axes=axes)

        # Axis labels:
        axes[0, 0].set_xlabel(r'$x_\odot$ ' + f'[{self.x_unit:latex_inline}]')
        axes[0, 0].set_ylabel(r'$y_\odot$ ' + f'[{self.x_unit:latex_inline}]')

        for ax in axes[1]:
            ax.set_xlabel(r'$v_{x, \odot}$ ' + f'[{self.v_unit:latex_inline}]')
        axes[1, 0].set_ylabel(r'$v_{y, \odot}$ ' + f'[{self.v_unit:latex_inline}]')

        anim = FuncAnimation(
            fig, animate_func, frames=len(self._wedge_iter),
            interval=20, blit=True, repeat=True, init_func=init_func)

        return anim
