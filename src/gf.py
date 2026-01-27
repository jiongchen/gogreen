import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import sph_harm_y

from src.material_func import gg_material_func
from src.radial_func import gg_radial_func

jax.config.update("jax_enable_x64", True)


class gg_regGF:
    def __init__(self, C_flat, maxl, eps, lebedev_order):
        self.C = gg_material_func(C_flat=C_flat, maxl=maxl, lebedev_order=lebedev_order)
        self.R = gg_radial_func(eps=eps, maxl=maxl)
        self.maxl = maxl

    def G(self, r, theta, phi):
        ret = jnp.zeros((r.shape[0], 3, 3))
        for l in range(0, self.maxl, 2):
            r_term = 1.0/(2*jnp.pi**2) * (1.0j**l) * self.R.Rlr(l, r)
            comp_l = jnp.zeros_like(ret)
            for m in range(-l, l+1):
                d_term = sph_harm_y(
                    jnp.full_like(theta, l, dtype=int),
                    jnp.full_like(theta, m, dtype=int),
                    theta,
                    phi,
                    n_max=self.maxl-1)
                comp_l += jnp.einsum('i,jk->ijk', r_term * d_term, self.C.Clm(l, m))            
            ret += comp_l
            print(f"[G at band l={l}] norm={jnp.linalg.norm(comp_l)}")

        return ret.real

    def graG(self, r, theta, phi):
        ret = jnp.zeros((r.shape[0], 3, 3, 3))
        for l in range(1, self.maxl, 2):
            r_term = 1.0/(2*jnp.pi**2) * (1.0j**l) * self.R.RRlr(l, r)
            comp_l = jnp.zeros_like(ret)
            for m in range(-l, l+1):
                d_term = sph_harm_y(
                    jnp.full_like(theta, l, dtype=int),
                    jnp.full_like(theta, m, dtype=int),
                    theta,
                    phi,
                    n_max=self.maxl-1)
                comp_l += jnp.einsum('i,jkl->ijkl', r_term*d_term, self.C.CClm(l, m))
            ret += comp_l
            print(f"[dG at band l={l}] norm={jnp.linalg.norm(comp_l)}")

        return ret.real