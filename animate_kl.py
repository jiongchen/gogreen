import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jax
import jax.numpy as jnp

from src.gf import gg_reg_gf

jax.config.update("jax_enable_x64", True)


def xyz_to_rtp(qx, qy, qz):
    """
    Cartesian (x, y, z) to Spherical (phi, theta)
    """
    r = jnp.sqrt(qx**2 + qy**2 + qz**2)    
    theta = jnp.acos(qz / r)
    phi = jnp.atan2(qy, qx)    
    return r, theta, phi


def orthotropic_material(E1, E2, E3, v12, v23, v13, G12, G23, G13):
    v21 = v12 * (E2 / E1)
    v32 = v23 * (E3 / E2)
    v31 = v13 * (E3 / E1)
    
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E1, 1/E2, 1/E3
    S[0, 1] = S[1, 0] = -v21/E2
    S[0, 2] = S[2, 0] = -v31/E3
    S[1, 2] = S[2, 1] = -v32/E3
    S[3, 3], S[4, 4], S[5, 5] = 1/G23, 1/G13, 1/G12
    
    return np.linalg.inv(S)


def tensor_to_voigt(C_ijkl):
    map_indices = {
        0: (0, 0),
        1: (1, 1),
        2: (2, 2),
        3: (1, 2), # yz
        4: (0, 2), # xz
        5: (0, 1)  # xy
    }
    
    C_voigt = np.zeros((6, 6))
    
    for i in range(6):
        for j in range(6):
            p, q = map_indices[i]
            r, s = map_indices[j]            
            C_voigt[i, j] = C_ijkl[p, q, r, s]
            
    return jnp.array(C_voigt)


def build_planar_grid(nx, ny, x_extent, y_extent):
    x = np.linspace(-x_extent, x_extent, nx)
    y = np.linspace(-y_extent, y_extent, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return X, Y, points


def green_surface(points, C_voigt, maxl, eps, lebedev_order):
    gf = gg_reg_gf(C_voigt=C_voigt, maxl=maxl, eps=eps, lebedev_order=lebedev_order)

    rtp = jnp.array(xyz_to_rtp(points[:, 0], points[:, 1], points[:, 2])).T    
    Gx = gf.G(rtp[:, 0], rtp[:, 1], rtp[:, 2])
    force = jnp.array([0.0, 0.0, 1.0])
    disp = jnp.einsum("ijk,k->ij", Gx, force)
    return np.array(disp[:, 2])/np.max(disp[:, 2])


def configure_axis(ax, X, Y, U):
    ax.clear()
    ax.plot_surface(X, Y, U, cmap="viridis", edgecolor="k", linewidth=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u_z")
    ax.set_title(f"Regularized Green's function surface")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(U.min(), U.max())
    ax.view_init(elev=35, azim=-145)


def main():
    n = 128
    extend = 4
    X, Y, points = build_planar_grid(n, n, extend, extend)

    # isotropic material
    E, nu = 1.0, 0.3
    mu = E/(2+2*nu)
    lam = E*nu/((1+nu)*(1-2*nu))
    delta = jnp.eye(3)
    C_tens = lam*jnp.einsum('ij,kl->ijkl', delta, delta) + mu*(jnp.einsum('ik,jl->ijkl', delta, delta) + jnp.einsum('il,jk->ijkl', delta, delta))
    C_voigt_iso = tensor_to_voigt(C_tens)

    # anisotropic material
    C_voigt_orth = orthotropic_material(
        E1=100, E2=1, E3=1,
        v12=0.3, v23=0.3, v13=0.3,
        G12=0.1, G23=0.1, G13=3
    )

    frames = 20
    ts = np.linspace(0, 1, frames)
    surfaces = [
        green_surface(
            points=points,
            C_voigt=(1.0-t)*C_voigt_iso + t*C_voigt_orth,
            maxl=20,
            eps=0.3,
            lebedev_order=25
        ).reshape(X.shape)
        for t in ts
    ]

    zmin = min(surface.min() for surface in surfaces)
    zmax = max(surface.max() for surface in surfaces)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        U = surfaces[frame_idx]
        ax.clear()
        ax.plot_surface(X, Y, U, cmap="viridis", edgecolor="k", linewidth=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u_z")
        ax.set_title("Isotropic to orthotropic GF")
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=35, azim=-125)
        return ()

    ani = FuncAnimation(fig, update, frames=frames, interval=140, blit=False)
    ani.save('ani_kl.gif', writer="pillow", fps=8)


if __name__ == "__main__":
    main()
