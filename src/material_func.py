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
def inv_elas_symbol(C_tensor, xi):
    C_ik = jnp.einsum('ijkl,j,l->ik', C_tensor, xi, xi)
    return jnp.linalg.inv(C_ik)


@jax.jit
def inv_elas_symbol_gra(C_tensor, xj):
    invC_ik = inv_elas_symbol(C_tensor, xj)
    return 1.0j*jnp.einsum('p,ij->pij', xj, invC_ik)

    
@jax.jit
def xyz_to_tp(qx, qy, qz):
    """
    Cartesian (x, y, z) to Spherical (phi, theta)
    """
    r = jnp.sqrt(qx**2 + qy**2 + qz**2)    
    theta = jnp.acos(qz / r)
    phi = jnp.atan2(qy, qx)
    return theta, phi


@jax.jit    
def tp_to_xyz(phi, theta):
    """
    unit Spherical (phi, theta) to (x, y, z)
    """
    qx = jnp.sin(theta)*jnp.cos(phi)
    qy = jnp.sin(theta)*jnp.sin(phi)
    qz = jnp.cos(theta)
    return qx, qy, qz


class gg_material_func:
    """
    material term
    """
    def __init__(self, C_voigt: jnp.ndarray, maxl: int, lebedev_order: int):
        self.maxl = maxl
        
        # compute lebedev quadrature
        qx, qw = lebedev_rule(lebedev_order)
        self.qx, self.qw = qx.T, qw
        print(f"lebedev order={lebedev_order}")
        print(f"qx shape={self.qx.shape}")
        print(f"sum(qw)={jnp.sum(qw)}, area={4*jnp.pi}")

        # polar coordinates of quadrature points
        thetas, phis = xyz_to_tp(self.qx[:,0], self.qx[:,1], self.qx[:,2])
        self.tp = jnp.column_stack([thetas, phis])

        # 3x3x3x3 tensor from voigt notattion
        mapping = jnp.array([
            [0, 5, 4],
            [5, 1, 3],
            [4, 3, 2]
        ])        
        i, j, k, l = jnp.meshgrid(
            jnp.arange(3),
            jnp.arange(3),
            jnp.arange(3),
            jnp.arange(3),
            indexing="ij"
        )
        ij = mapping[i, j]
        kl = mapping[k, l]
        self.C_tensor = C_voigt[ij, kl]

        # precompute material terms
        self.C_lm = {}
        self.dC_lm = {}
        self._precompute()

    def _intS_CijYlm(self, l, m):
        Cinv_at_q = vmap(inv_elas_symbol, in_axes=(None, 0))(self.C_tensor, self.qx)

        theta_vec = self.tp[:, 0]
        phi_vec = self.tp[:, 1]
        l_vec = jnp.full_like(theta_vec, l, dtype=int)
        m_vec = jnp.full_like(phi_vec, m, dtype=int)
        sh_at_q = jnp.conj(sph_harm_y(l_vec, m_vec, theta_vec, phi_vec, n_max=self.maxl-1))

        ret = jnp.einsum('i,ijk,i->jk', self.qw, Cinv_at_q, sh_at_q)
        return ret

    def _intS_xiCijYlm(self, l, m):
        xiCinv_at_q = vmap(inv_elas_symbol_gra, in_axes=(None, 0))(self.C_tensor, self.qx)

        theta_vec = self.tp[:, 0]
        phi_vec = self.tp[:, 1]
        l_vec = jnp.full_like(theta_vec, l, dtype=int)
        m_vec = jnp.full_like(phi_vec, m, dtype=int)
        sh_at_q = jnp.conj(sph_harm_y(l_vec, m_vec, theta_vec, phi_vec, n_max=self.maxl-1))

        ret = jnp.einsum('i,ijkl,i->jkl', self.qw, xiCinv_at_q, sh_at_q)
        return ret

    def _precompute(self):
        for l in range(0, self.maxl):
            for m in range(-l, l+1):
                self.C_lm[(l, m)] = self._intS_CijYlm(l, m)

        for l in range(0, self.maxl):
            for m in range(-l, l+1):
                self.dC_lm[(l, m)] = self._intS_xiCijYlm(l, m)

    def Clm(self, l, m):
        assert l >= 0 and l < self.maxl
        assert -l <= m and m <= l
        return self.C_lm[(l, m)]

    def CClm(self, l, m):
        assert l >= 0 and l < self.maxl
        assert -l <= m and m <= l
        return self.dC_lm[(l, m)]