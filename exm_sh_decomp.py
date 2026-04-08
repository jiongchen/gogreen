import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from matplotlib.animation import FuncAnimation

from src.gf import gg_reg_gf

jax.config.update("jax_enable_x64", True)


# -1 is for all-band deformation
BANDS = [-1, 0, 2, 4, 6, 8]
CMAPS = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']


def xyz_to_rtp(qx, qy, qz):
    """
    Cartesian (x, y, z) to spherical (r, theta, phi).

    The origin is handled explicitly because the angular coordinates are
    undefined there, while the regularized kernel remains finite.
    """
    r = jnp.sqrt(qx**2 + qy**2 + qz**2)
    safe_r = jnp.where(r > 1.0e-12, r, 1.0)
    theta = jnp.where(r > 1.0e-12, jnp.acos(qz / safe_r), 0.0)
    phi = jnp.where(r > 1.0e-12, jnp.atan2(qy, qx), 0.0)
    return r, theta, phi


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range) / 2.0

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)


def orthotropic_material(E1, E2, E3, v12, v23, v13, G12, G23, G13):
    v21 = v12 * (E2 / E1)
    v32 = v23 * (E3 / E2)
    v31 = v13 * (E3 / E1)

    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[0, 1] = S[1, 0] = -v21 / E2
    S[0, 2] = S[2, 0] = -v31 / E3
    S[1, 2] = S[2, 1] = -v32 / E3
    S[3, 3], S[4, 4], S[5, 5] = 1 / G23, 1 / G13, 1 / G12
    return jnp.array(np.linalg.inv(S))


def band_deformation(gf, l, rtp, force):
    if l != -1:
        # each band
        Gx_l = gf.G_l(l, rtp[:, 0], rtp[:, 1], rtp[:, 2])
        disp_l = jnp.einsum("ijk,k->ij", Gx_l, force)
    else:
        # all bands
        Gx = gf.G(rtp[:, 0], rtp[:, 1], rtp[:, 2])
        disp_l = jnp.einsum("ijk,k->ij", Gx, force)

    return np.array(disp_l)
    

if __name__ == "__main__":
    nx, ny = 64, 64
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)

    X, Y = np.meshgrid(x, y)
    print(X.shape, Y.shape)
    points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    rtp = jnp.array(xyz_to_rtp(points[:, 0], points[:, 1], points[:, 2])).T
    print(rtp.shape)

    C_voigt = orthotropic_material(
        E1=100, E2=1, E3=1,
        v12=0.3, v23=0.3, v13=0.3,
        G12=0.1, G23=0.1, G13=10
    )
    gf = gg_reg_gf(C_voigt=C_voigt, maxl=max(BANDS) + 1, eps=0.3, lebedev_order=35)

    force = jnp.array([0.0, 0.0, 6.0])
    deformations = {l: band_deformation(gf, l, rtp, force) for l in BANDS}

    surfaces = {}
    for l, disp in deformations.items():
        surfaces[l] = {
            "x": (points[:, 0] + disp[:, 0]).reshape(X.shape),
            "y": (points[:, 1] + disp[:, 1]).reshape(X.shape),
            "z": (points[:, 2] + disp[:, 2]).reshape(X.shape),
        }

    all_x = np.concatenate([surfaces[l]["x"].ravel() for l in BANDS])
    all_y = np.concatenate([surfaces[l]["y"].ravel() for l in BANDS])
    all_z = np.concatenate([surfaces[l]["z"].ravel() for l in BANDS])
    xyz_limits = (
        (float(all_x.min()), float(all_x.max())),
        (float(all_y.min()), float(all_y.max())),
        (float(all_z.min()), float(all_z.max())),
    )

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 10),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    axes = axes.ravel()

    def config_ax(ax, l, azim, color):
        ax.clear()
        surf = surfaces[l]
        ax.plot_surface(
            surf["x"],
            surf["y"],
            surf["z"],
            cmap=color,
            edgecolor="k",
            linewidth=0.1,
            antialiased=True,
            rstride=1,
            cstride=1
        )
        if l == -1:
            ax.set_title("Total deformation")
        else:
            ax.set_title(f"Band l={l}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(*xyz_limits[0])
        ax.set_ylim(*xyz_limits[1])
        ax.set_zlim(*xyz_limits[2])
        ax.view_init(elev=30, azim=azim)
        set_axes_equal(ax)

    frames = 72
    azimuths = np.linspace(-120.0, 240.0, frames)

    def update(frame_idx):
        azim = azimuths[frame_idx]
        artists = []
        for ax, l, c in zip(axes, BANDS, CMAPS):
            config_ax(ax, l, azim, c)
            artists.append(ax)

        return tuple(artists)

    os.makedirs("result", exist_ok=True)
    ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    ani.save("result/sh_bands.gif", writer="pillow", fps=15)
