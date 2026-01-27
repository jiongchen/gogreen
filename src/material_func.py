import jax
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
from scipy.integrate import lebedev_rule
import numpy as np
from jax.scipy.special import sph_harm_y
from functools import partial
from jax import jit

jax.config.update("jax_enable_x64", True)


@jax.jit
def inv_elas_symbol(C_vec, xi):
    C_tensor = C_vec.reshape((3, 3, 3, 3))
    C_ik = jnp.einsum('ijkl,j,l->ik', C_tensor, xi, xi)
    return jnp.linalg.inv(C_ik)

    
@jax.jit
def inv_elas_symbol_gra(C_vec, xj):
    C_tensor = C_vec.reshape((3, 3, 3, 3))
    C_ik = jnp.einsum('ijkl,j,l->ik', C_tensor, xj, xj)
    invC_ik = jnp.linalg.inv(C_ik)
    return 1.0j*jnp.einsum('j,ik->jik', xj, invC_ik)

    
@jax.jit
def xyz_to_tp(qx, qy, qz):
    """
    Cartesian (x, y, z) to Spherical (phi, theta)
    """
    r = jnp.sqrt(qx**2 + qy**2 + qz**2)    
    theta = jnp.acos(qz / r)
    phi = jnp.atan2(qy, qx)    
    return phi, theta


@jax.jit
def tp_to_xyz(phi, theta):
    """
    Spherical (phi, theta) to (x, y, z)
    """
    qx = jnp.sin(theta)*jnp.cos(phi)
    qy = jnp.sin(theta)*jnp.sin(phi)
    qz = jnp.cos(theta)
    return qx, qy, qz


class gg_material_func:
    def __init__(
            self,            
            C_flat: jnp.ndarray,
            maxl: int,
            num_quadr: int
    ):
        # flattened material tensor
        self.C_flat = C_flat
        self.maxl = maxl
        
        # compute lebedev quadrature
        qx, qw = lebedev_rule(num_quadr)
        self.qx, self.qw = qx.T, qw
        print(f"qx shape={self.qx.shape}")
        print(f"sum(qw)={jnp.sum(qw)}")

        # polar coordinates of quadrature points
        phis, thetas = xyz_to_tp(self.qx[:,0], self.qx[:,1], self.qx[:,2])
        self.tp = jnp.column_stack([phis, thetas])

        self.C_lm = {}
        self.dC_lm = {}
        self._precompute()

    def _intS_CijYlm(self, l, m):
        Cinv_at_q = vmap(inv_elas_symbol, in_axes=(None, 0))(self.C_flat, self.qx)

        theta_vec = self.tp[:, 0]
        phi_vec = self.tp[:, 1]
        l_vec = jnp.full_like(theta_vec, l, dtype=int)
        m_vec = jnp.full_like(phi_vec, m, dtype=int)
        sh_at_q = jnp.conj(sph_harm_y(l_vec, m_vec, theta_vec, phi_vec))

        ret = jnp.einsum('i,ijk,i->jk', self.qw, Cinv_at_q, sh_at_q)
        return ret

    def _intS_xiCijYlm(self, l, m):
        xiCinv_at_q = vmap(inv_elas_symbol_gra, in_axes=(None, 0))(self.C_flat, self.qx)

        theta_vec = self.tp[:, 0]
        phi_vec = self.tp[:, 1]
        l_vec = jnp.full_like(theta_vec, l, dtype=int)
        m_vec = jnp.full_like(phi_vec, m, dtype=int)
        sh_at_q = jnp.conj(sph_harm_y(l_vec, m_vec, theta_vec, phi_vec))

        ret = jnp.einsum('i,ijkl,i->jkl', self.qw, xiCinv_at_q, sh_at_q)
        return ret

    def _precompute(self):
        for l in range(0, self.maxl, 2):
            for m in range(-l, l):
                self.C_lm[(l, m)] = self._intS_CijYlm(l, m)

        for l in range(1, self.maxl, 2):
            for m in range(-l, l):
                self.dC_lm[(l, m)] = self._intS_xiCijYlm(l, m)

    def Clm(self, l, m):
        assert l >= 0 and l < self.maxl and l%2 == 0
        assert -l <= m and m <= l
        return self.C_lm[(l, m)]

    def CClm(self, l, m):
        assert l >= 1 and l < self.maxl and l%2 == 1
        assert -l <= m and m <= l
        return self.dC_lm[(l, m)]


if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    mu = 1.0
    lam = 1.0
    
    def delta(i, j): return 1.0 if i == j else 0.0

    C_tens = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C_tens[i,j,k,l] = lam * delta(i,j)*delta(k,l) + mu * (delta(i,k)*delta(j,l) + delta(i,l)*delta(j,k))
    
    C_flat = jnp.array(C_tens.flatten())
    
    mat_func = gg_material_func(C_flat, maxl=20, num_quadr=11)
    print(mat_func.Clm(2, 0))
    print(mat_func.CClm(1, 0))
    
    x, y, z = mat_func.qx[:,0], mat_func.qx[:,1], mat_func.qx[:,2] 
    scatter = ax.scatter(x, y, z, c='red', cmap='viridis', s=50)

    rx, ry, rz = tp_to_xyz(mat_func.tp[:, 0], mat_func.tp[:, 1])
    scatter = ax.scatter(x, y, z, c='blue', cmap='viridis', s=30)

    # Labels and title
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Scatter Plot (Matplotlib)')

    # Add a color bar
    plt.colorbar(scatter, ax=ax, label='Intensity')
    plt.show()