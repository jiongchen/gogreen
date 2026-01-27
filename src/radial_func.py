import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import gamma
from functools import partial
import matplotlib.pyplot as plt
from jax.scipy.special import hyp1f1

jax.config.update("jax_enable_x64", True)


class gg_radial_func:
    """
    The singularity is smoothed out with exp impulses
    """
    def __init__(self, eps: float, maxl: int):
        """
        eps: epsilon for regularizing GF
        maxl: maximum bands
        """
        self.eps_ = eps
        self.SQRT_PI = jnp.sqrt(jnp.pi)
        self.MAXL = maxl
        
    def Rlr(self, l, r):
        gl = (self.eps_**(-1.0 - l)) * gamma((1.0 + l) / 2.0) 
        gaml = 1.0 / gamma(1.5 + l)
        arg_1f1 = -(r * r) / (self.eps_ * self.eps_)
        hf11 = hyp1f1((1.0 + l) / 2.0, 1.5 + l, arg_1f1)

        return self.SQRT_PI/2.0 * (r**l) * gl * hf11 * gaml

    def RRlr(self, l, r):        
        gl = (self.eps_**(-2.0 - l)) * gamma(1.0 + l / 2.0)
        gaml = 1.0 / gamma(1.5 + l)
        arg_1f1 = -(r * r) / (self.eps_ * self.eps_)
        hf11 = hyp1f1(1.0 + l / 2.0, 1.5 + l, arg_1f1)
        
        return self.SQRT_PI * (r**l) * gl * hf11 * gaml

        
if __name__ == "__main__":
    MAXL = 20
    EPS = 0.2
    R_POINTS = jnp.linspace(0, 1, 500)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # for GF, only even bands
    L_VALUES = [0, 2, 4, 6, 8]    
    radial_func = gg_radial_func(eps=EPS, maxl=MAXL)
    results_vector = {}    
    for l in L_VALUES:
        rl = radial_func.Rlr(l, R_POINTS)
        results_vector[l] = rl

    # for derivatives of GF, only odd bands
    L_VALUES = [1, 3, 5, 7, 9]
    radial_func = gg_radial_func(eps=EPS, maxl=MAXL)
    results_matrix = {}
    for l in L_VALUES:
        Rl = radial_func.RRlr(l, R_POINTS)
        results_matrix[l] = Rl

    for l, y in results_vector.items():
        axes[0].plot(R_POINTS, y, label=f'l={l}', linewidth=2)
    axes[0].set_title(f"$R_l(r), \epsilon$={EPS}")
    axes[0].set_xlabel("$r$")
    axes[0].legend()
                
    for l, y in results_matrix.items():
        axes[1].plot(R_POINTS, y, label=f'l={l}', linewidth=2)
    axes[1].set_title(f"$R_l(r), \epsilon$={EPS}")
    axes[1].set_xlabel("$r$")
    axes[1].legend()

    plt.tight_layout()
    plt.show()