import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import matplotlib.pyplot as plt
from jax.scipy.special import sph_harm_y
import numpy as np

from src.material_func import gg_material_func
from src.radial_func import gg_radial_func

jax.config.update("jax_enable_x64", True)


class gg_regGF:
    def __init__(self, C_flat, maxl, eps, numQ):
        self.C = gg_material_func(C_flat=C_flat, maxl=maxl, num_quadr=numQ)
        self.R = gg_radial_func(eps=eps, maxl=maxl)
        self.maxl = maxl

    def G(self, r, theta, phi):
        ret = jnp.zeros((r.shape[0], 3, 3))
        for l in range(0, self.maxl, 2):
            r_term = 1.0/(2*jnp.pi**2) * (1.0j**l) * self.R.Rlr(l, r)
            for m in range(-l, l+1):
                d_term = sph_harm_y(
                    jnp.full_like(theta, l, dtype=int),
                    jnp.full_like(theta, m, dtype=int),
                    theta,
                    phi,
                    n_max=self.maxl-1)
                ret += jnp.einsum('i,jk->ijk', r_term * d_term, self.C.Clm(l, m))

        return ret.real

    def graG(self, r, theta, phi):
        ret = jnp.zeros((r.shape[0], 3, 3, 3))
        for l in range(1, self.maxl, 2):
            r_term = 1.0/(2*jnp.pi**2) * (1.0j**l) * self.R.RRlr(l, r)
            for m in range(-l, l+1):
                d_term = sph_harm_y(
                    jnp.full_like(theta, l, dtype=int),
                    jnp.full_like(theta, m, dtype=int),
                    theta,
                    phi,
                    n_max=self.maxl-1)
                ret += jnp.einsum('i,jkl->ijkl', r_term*d_term, self.C.CClm(l, m))

        return ret.real


if __name__ == "__main__":
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
    GF = gg_regGF(C_flat=C_flat, maxl=10, eps=0.2, numQ=13)

    # r_values = jnp.linspace(0, 5, 500)
    # theta_values = jnp.zeros((500,))
    # phi_values = jnp.zeros((500,))
    # print(r_values.shape, theta_values.shape, phi_values.shape)
    # y_values = GF.G(r_values, theta_values, phi_values)

    # plt.figure(figsize=(8, 5))
    # plt.plot(r_values, y_values, label='$f(r)$', color='blue', linewidth=2)

    # plt.title("Radial Function Plot")
    # plt.xlabel("Radius $r$")
    # plt.ylabel("$f(r)$")
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()

    # plt.show()    
    