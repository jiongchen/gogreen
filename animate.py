import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from src.gf import gg_reg_gf
from matplotlib.animation import FuncAnimation

jax.config.update("jax_enable_x64", True)


def xyz_to_rtp(qx, qy, qz):
    """
    Cartesian (x, y, z) to Spherical (phi, theta)
    """
    r = jnp.sqrt(qx**2 + qy**2 + qz**2)    
    theta = jnp.acos(qz / r)
    phi = jnp.atan2(qy, qx)    
    return r, theta, phi

    
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

    
if __name__ == "__main__":
    nx, ny = 30, 30
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-0.6, 0.6, ny)

    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()]).T
    Z = np.zeros(points.shape[0])
    points = np.column_stack([points, Z])

    # catesian to spherical coords
    rtp = jnp.array(xyz_to_rtp(points[:, 0], points[:, 1], points[:, 2])).T

    # isotropic material
    E, nu = 1.0, 0.3
    mu = E/(2+2*nu)
    lam = E*nu/((1+nu)*(1-2*nu))
    print(f"lambda={lam}, mu={mu}")
    delta = jnp.eye(3)
    C_tens = lam*jnp.einsum('ij,kl->ijkl', delta, delta) + mu*(jnp.einsum('ik,jl->ijkl', delta, delta) + jnp.einsum('il,jk->ijkl', delta, delta))
    C_voigt = tensor_to_voigt(C_tens)
    GF = gg_reg_gf(C_voigt=C_voigt, maxl=8, eps=0.3, lebedev_order=11)
    dGx = GF.graG(rtp[:, 0], rtp[:, 1], rtp[:, 2])

    # orthotropic material
    C_voigt_ort = orthotropic_material(
        E1=10, E2=1, E3=1,
        v12=0.3, v23=0.3, v13=0.3, 
        G12=0.6, G23=0.6, G13=0.6
    )
    GF_ort = gg_reg_gf(C_voigt=C_voigt_ort, maxl=30, eps=0.3, lebedev_order=11)
    dGx_ort = GF_ort.graG(rtp[:, 0], rtp[:, 1], rtp[:, 2])

    # plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True, subplot_kw={'projection': '3d'})
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[0, 2]
    ax4 = axes[1, 0]
    ax5 = axes[1, 1]
    ax6 = axes[1, 2]

    frames = 100
    p_values = np.linspace(0, 0.4, frames)
    s_values = np.linspace(0, 1.5, frames)
    t_values = np.linspace(0, 0.5, frames)

    def config_ax(ax):
        ax.clear()
        ax.view_init(elev=90, azim=-90)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-0.1, 0.1)

    def update(frame):
        # pinching
        s = p_values[frame]    
        F = jnp.array([
            [-s, 0, 0],
            [ 0, s, 0],
            [ 0, 0, 0]
        ])
        disp_m = jnp.einsum('ijkl,lj->ik', dGx, F)    
        new_X = (points[:, 0] + disp_m[:, 0]).reshape(X.shape)
        new_Y = (points[:, 1] + disp_m[:, 1]).reshape(X.shape)
        new_Z = (points[:, 2] + disp_m[:, 2]).reshape(X.shape)        
        config_ax(ax1)
        surf1 = ax1.plot_surface(new_X, new_Y, new_Z, color='violet', edgecolor='orange', linewidth=0.7, alpha=0.0)    
        ax1.set_title('Pinching')
        set_axes_equal(ax1)

        disp_m = jnp.einsum('ijkl,lj->ik', dGx_ort, F)    
        new_X = (points[:, 0] + disp_m[:, 0]).reshape(X.shape)
        new_Y = (points[:, 1] + disp_m[:, 1]).reshape(X.shape)
        new_Z = (points[:, 2] + disp_m[:, 2]).reshape(X.shape)        
        config_ax(ax4)
        surf4 = ax4.plot_surface(new_X, new_Y, new_Z, color='violet', edgecolor='red', linewidth=0.7, alpha=0.0)
        set_axes_equal(ax4)
        
        # scaling
        s = -s_values[frame]
        F = s*jnp.eye(3)        
        disp_s = jnp.einsum('ijkl,lj->ik', dGx, F)
        new_X = (points[:, 0] + disp_s[:, 0]).reshape(X.shape)
        new_Y = (points[:, 1] + disp_s[:, 1]).reshape(X.shape)
        new_Z = (points[:, 2] + disp_s[:, 2]).reshape(X.shape)
        config_ax(ax2)
        surf2 = ax2.plot_surface(new_X, new_Y, new_Z, color='violet', edgecolor='teal', linewidth=0.7, alpha=0.0)    
        ax2.set_title('Scaling')
        set_axes_equal(ax2)

        disp_s = jnp.einsum('ijkl,lj->ik', dGx_ort, F)
        new_X = (points[:, 0] + disp_s[:, 0]).reshape(X.shape)
        new_Y = (points[:, 1] + disp_s[:, 1]).reshape(X.shape)
        new_Z = (points[:, 2] + disp_s[:, 2]).reshape(X.shape)
        config_ax(ax5)
        surf5 = ax5.plot_surface(new_X, new_Y, new_Z, color='violet', edgecolor='blue', linewidth=0.7, alpha=0.0)
        set_axes_equal(ax5)        

        # twisting
        s = t_values[frame]
        F = jnp.array([
            [0, s, 0],
            [-s, 0, 0],
            [0, 0, 0]
        ])
        disp_t = jnp.einsum('ijkl,lj->ik', dGx, F)
        new_X = (points[:, 0] + disp_t[:, 0]).reshape(X.shape)
        new_Y = (points[:, 1] + disp_t[:, 1]).reshape(X.shape)
        new_Z = (points[:, 2] + disp_t[:, 2]).reshape(X.shape)
        config_ax(ax3)
        surf3 = ax3.plot_surface(new_X, new_Y, new_Z, color='violet', edgecolor='purple', linewidth=0.7, alpha=0.0)    
        ax3.set_title('Twisting')
        set_axes_equal(ax3)

        disp_t = jnp.einsum('ijkl,lj->ik', dGx_ort, F)
        new_X = (points[:, 0] + disp_t[:, 0]).reshape(X.shape)
        new_Y = (points[:, 1] + disp_t[:, 1]).reshape(X.shape)
        new_Z = (points[:, 2] + disp_t[:, 2]).reshape(X.shape)
        config_ax(ax6)
        surf6 = ax6.plot_surface(new_X, new_Y, new_Z, color='violet', edgecolor='black', linewidth=0.7, alpha=0.0)
        set_axes_equal(ax6) 

        return surf1, surf2, surf3, surf4, surf5, surf6
    
    ani = FuncAnimation(fig, update, frames=len(s_values), interval=5, blit=False)
    ani.save('result.gif', writer='pillow', fps=30)