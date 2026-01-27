import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mpl_toolkits.mplot3d import Axes3D
import jax
from src.gf import gg_regGF


jax.config.update("jax_enable_x64", True)


def xyz_to_rtp(qx, qy, qz):
    """
    Cartesian (x, y, z) to Spherical (phi, theta)
    """
    r = jnp.sqrt(qx**2 + qy**2 + qz**2)    
    theta = jnp.acos(qz / r)
    phi = jnp.atan2(qy, qx)    
    return r, phi, theta

    
def set_axes_equal(ax):
    """
    Set 3D plot axes to equal scale.

    Works with any 3D plot: scatter, plot_trisurf, wireframe, etc.
    """
    # Extract current limits
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate ranges
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range) / 2.0

    # Calculate midpoints
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    # Set new limits centered at midpoints with max_range
    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)
    

if __name__ == "__main__":
    nx, ny = 30, 30
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)

    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    Z = np.zeros(points.shape[0])
    points = np.column_stack([points, Z])

    # catesian to spherical coords
    rtp = jnp.array(xyz_to_rtp(points[:, 0], points[:, 1], points[:, 2])).T

    # material
    E, nu = 1.0, 0.3
    mu = E/(2+2*nu)
    lam = E*nu/((1+nu)*(1-2*nu))

    # isotropic tensor
    delta = jnp.eye(3)
    C_tens = lam*jnp.einsum('ij,kl->ijkl', delta, delta) + mu*(jnp.einsum('ik,jl->ijkl', delta, delta) + jnp.einsum('il,kj->ijkl', delta, delta))
    C_flat = jnp.array(C_tens.flatten())

    # GF
    GF = gg_regGF(C_flat=C_flat, maxl=20, eps=0.3, lebedev_order=83)

    # vector impulse
    s = 3
    Gx = GF.G(rtp[:, 0], rtp[:, 1], rtp[:, 2])
    f = jnp.array([0, 0, s])
    disp_v = jnp.einsum('ijk,k->ij', Gx, f)

    # pinching
    s = -0.2
    dGx = GF.graG(rtp[:, 0], rtp[:, 1], rtp[:, 2])
    F = jnp.array([
        [-s, 0, 0],
        [0, s, 0],
        [0, 0, 0]
    ])
    disp_m = jnp.einsum('ijkl,lj->ik', dGx, F)

    # scaling
    s = -1
    F = s*jnp.eye(3)
    disp_s = jnp.einsum('ijkl,lj->ik', dGx, F)

    # twisting
    s = -0.3
    F = jnp.array([
        [0, s, 0],
        [-s, 0, 0],
        [0, 0, 0]
    ])
    disp_t = jnp.einsum('ijkl,lj->ik', dGx, F)
    
    # plot
    fig = plt.figure(figsize=(32, 8))

    ax0 = fig.add_subplot(141, projection='3d')
    ax0.plot_surface(
        (points[:, 0] + disp_v[:, 0]).reshape(X.shape),
        (points[:, 1] + disp_v[:, 1]).reshape(X.shape),
        (points[:, 2] + disp_v[:, 2]).reshape(X.shape),        
        color='teal',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.0
    )
    ax0.set_title('vector impulse')
    set_axes_equal(ax0)

    ax1 = fig.add_subplot(142, projection='3d')
    ax1.view_init(elev=90, azim=-90)        
    ax1.plot_surface(
        (points[:, 0] + disp_m[:, 0]).reshape(X.shape),
        (points[:, 1] + disp_m[:, 1]).reshape(X.shape),
        (points[:, 2] + disp_m[:, 2]).reshape(X.shape),
        color='violet',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.0
    )
    ax1.set_title('pinching')
    set_axes_equal(ax1)

    ax2 = fig.add_subplot(143, projection='3d')
    ax2.view_init(elev=90, azim=-90)    
    ax2.plot_surface(
        (points[:, 0] + disp_s[:, 0]).reshape(X.shape),
        (points[:, 1] + disp_s[:, 1]).reshape(X.shape),
        (points[:, 2] + disp_s[:, 2]).reshape(X.shape),
        color='violet',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.0
    )
    ax2.set_title('scaling')
    set_axes_equal(ax2)

    ax3 = fig.add_subplot(144, projection='3d')
    ax3.view_init(elev=90, azim=-90)
    ax3.plot_surface(
        (points[:, 0] + disp_t[:, 0]).reshape(X.shape),
        (points[:, 1] + disp_t[:, 1]).reshape(X.shape),
        (points[:, 2] + disp_t[:, 2]).reshape(X.shape),
        color='violet',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.0
    )
    ax3.set_title('twisting')
    set_axes_equal(ax3)

    plt.show()
    plt.savefig('result.png')