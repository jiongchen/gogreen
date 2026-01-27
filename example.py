import trimesh
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mpl_toolkits.mplot3d import Axes3D
import jax
from src.gf import gg_regGF
from scipy.spatial import Delaunay

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
    points = np.vstack([X.ravel(), Y.ravel()]).T  # shape (nx*ny, 2)

    tri = Delaunay(points)
    faces = tri.simplices 
    Z = np.zeros(points.shape[0])
    points = np.column_stack([points, Z])

    # catesian to spherical coords
    rtp = jnp.array(xyz_to_rtp(points[:, 0], points[:, 1], points[:, 2])).T

    # material
    E, nu = 1.0, 0.4
    mu = E/(2+2*nu)
    lam = E*nu/((1+nu)*(1-2*nu))
    
    def delta(i, j):
        return 1.0 if i == j else 0.0

    C_tens = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C_tens[i,j,k,l] = lam * delta(i,j)*delta(k,l) + mu * (delta(i,k)*delta(j,l) + delta(i,l)*delta(j,k))    
    C_flat = jnp.array(C_tens.flatten())

    # GF
    GF = gg_regGF(C_flat=C_flat, maxl=10, eps=0.5, numQ=13)

    # vector impulse
    s = 3
    Gx = GF.G(rtp[:, 0], rtp[:, 1], rtp[:, 2])
    f = jnp.array([0, 0, s])
    disp = jnp.einsum('ijk,k->ij', Gx, f)

    # matrix impulse
    s = -1
    dGx = GF.graG(rtp[:, 0], rtp[:, 1], rtp[:, 2])
    F = jnp.array([
        [-s, 0, 0],
        [0, s, 0],
        [0, 0, 0]
    ])
    disp = jnp.einsum('ijkl,jl->ik', dGx, F)
    print(dGx.shape)
    
    # plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        points[:, 0] + disp[:, 0],
        points[:, 1] + disp[:, 1],
        points[:, 2] + disp[:, 2],
        triangles=faces,
        color='lightblue',
        edgecolor='gray',
        linewidth=0.5,
        alpha=0.8
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)

    plt.show()