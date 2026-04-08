import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jax
import jax.numpy as jnp

from src.gf import gg_reg_gf

jax.config.update("jax_enable_x64", True)


BANDS = [-1, 1, 3, 5, 7, 9]
CMAPS = ["Greys", "Purples", "Blues", "Greens", "Oranges", "Reds"]


def xyz_to_rtp(qx, qy, qz):
    r = jnp.sqrt(qx**2 + qy**2 + qz**2)
    safe_r = jnp.where(r > 1.0e-12, r, 1.0)
    theta = jnp.where(r > 1.0e-12, jnp.acos(qz / safe_r), 0.0)
    phi = jnp.where(r > 1.0e-12, jnp.atan2(qy, qx), 0.0)
    return r, theta, phi


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


def build_regular_grid(nx, ny, nz, extent):
    x = np.linspace(-extent, extent, nx)
    y = np.linspace(-extent, extent, ny)
    z = np.linspace(-extent, extent, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return X, Y, Z, points


def scaling_tensor(s):
    return s*jnp.eye(3)            


def band_deformation(gf, l, rtp, scale):
    if l == -1:
        grad_g = gf.graG(rtp[:, 0], rtp[:, 1], rtp[:, 2])
    else:
        grad_g = gf.graG_l(l, rtp[:, 0], rtp[:, 1], rtp[:, 2])

    disp = jnp.einsum("pijk,jk->pi", grad_g, scale)
    return np.array(disp)


def set_axes_equal(ax, xyz_limits):
    x_limits, y_limits, z_limits = xyz_limits
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range) / 2.0

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def draw_box(ax, limits):
    x0, x1 = limits[0]
    y0, y1 = limits[1]
    z0, z1 = limits[2]

    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i0, i1 in edges:
        pts = corners[[i0, i1]]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color="0.75",
            linewidth=0.7,
            alpha=0.6,
        )


def main():
    nx = ny = nz = 32
    extent = 1.0
    maxl = max(BANDS) + 1
    scale_strength = -0.2

    _, _, _, points = build_regular_grid(nx, ny, nz, extent)
    rtp = jnp.array(xyz_to_rtp(points[:, 0], points[:, 1], points[:, 2])).T

    C_voigt = orthotropic_material(
        E1=100,
        E2=1,
        E3=1,
        v12=0.3,
        v23=0.3,
        v13=0.3,
        G12=0.1,
        G23=0.1,
        G13=10,
    )
    gf = gg_reg_gf(C_voigt=C_voigt, maxl=maxl, eps=0.3, lebedev_order=35)

    scale = scaling_tensor(scale_strength)
    deformations = {
        l: band_deformation(gf, l, rtp, scale) for l in BANDS
    }

    deformed_points = {l: points + deformations[l] for l in BANDS}
    magnitudes = {l: np.linalg.norm(deformations[l], axis=1) for l in BANDS}

    all_xyz = np.concatenate([deformed_points[l] for l in BANDS], axis=0)
    xyz_limits = (
        (float(all_xyz[:, 0].min()), float(all_xyz[:, 0].max())),
        (float(all_xyz[:, 1].min()), float(all_xyz[:, 1].max())),
        (float(all_xyz[:, 2].min()), float(all_xyz[:, 2].max())),
    )
    base_limits = ((-extent, extent), (-extent, extent), (-extent, extent))

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 10),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    axes = axes.ravel()

    def config_ax(ax, l, azim, cmap):
        ax.clear()
        pts = deformed_points[l]
        vals = magnitudes[l]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=vals,
            cmap=cmap,
            s=9,
            alpha=0.6,
            linewidths=0.0,
        )
        draw_box(ax, base_limits)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=24, azim=azim)
        set_axes_equal(ax, xyz_limits)

        if l == -1:
            title = "Total deformation"
        else:
            title = f"Band l={l}"
        ax.set_title(title)

    frames = 72
    azimuths = np.linspace(-135.0, 225.0, frames)

    def update(frame_idx):
        azim = azimuths[frame_idx]
        artists = []
        for ax, l, cmap in zip(axes, BANDS, CMAPS):
            config_ax(ax, l, azim, cmap)
            artists.append(ax)
        return tuple(artists)

    os.makedirs("result", exist_ok=True)
    ani = FuncAnimation(fig, update, frames=frames, interval=90, blit=False)
    ani.save("result/sh_bands_gra.gif", writer="pillow", fps=15)


if __name__ == "__main__":
    main()
