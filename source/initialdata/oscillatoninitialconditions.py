"""
Set the initial conditions for all the variables for an oscillaton.

See further details in https://github.com/GRChombo/engrenage/wiki/Running-the-oscillaton-example.
"""

import numpy as np  # Used only to read the CSV files
import jax.numpy as jnp

from bssn.tensoralgebra import *
from core.grid import Grid
from bssn.bssnstatevariables import *
from backgrounds.sphericalbackground import *


def get_initial_state(grid: Grid, background):

    assert grid.NUM_VARS == 14, "NUM_VARS not correct for bssn + scalar field"

    # For readability
    r = grid.r
    N = grid.num_points
    dr = grid.min_dr
    NUM_VARS = grid.NUM_VARS

    # Load the data, then convert it to JAX arrays.
    grr0_data = jnp.asarray(
        np.loadtxt("../source/initialdata/oscillaton/grr0.csv")
    )
    lapse0_data = jnp.asarray(
        np.loadtxt("../source/initialdata/oscillaton/lapse0.csv")
    )
    v0_data = jnp.asarray(
        np.loadtxt("../source/initialdata/oscillaton/v0.csv")
    )

    length = grr0_data.size
    grr0_data = jnp.concatenate((jnp.flip(grr0_data), grr0_data[1:length]))
    lapse0_data = jnp.concatenate((jnp.flip(lapse0_data), lapse0_data[1:length]))
    v0_data = jnp.concatenate((jnp.flip(v0_data), v0_data[1:length]))

    # Set up the grid in the radial direction in areal polar coordinates.
    dR = 0.01
    assert dR < dr, "your dr is smaller than the oscillaton data, use fewer points!"
    R = jnp.linspace(-dR * (length - 1), dR * (length - 1), num=length * 2 - 1)

    # jnp.interp performs linear interpolation, like scipy's interp1d default.
    # Check the range because jnp.interp otherwise clamps outside it.
    assert (
        jnp.min(r) >= R[0] and jnp.max(r) <= R[-1]
    ), "Grid points lie outside the oscillaton data range."

    # Set the non-zero scalar field values and the lapse and spatial metric.
    v = jnp.interp(r, R, v0_data)
    lapse = jnp.interp(r, R, lapse0_data)
    grr = jnp.interp(r, R, grr0_data)

    gtt_over_r2 = 1.0
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta

    # Work out the rescaled quantities.
    # Note sign error in Baumgarte eqn (2), conformal factor.
    phi = (1.0 / 12.0) * jnp.log(phys_gamma_over_r4sin2theta)
    em4phi = jnp.exp(-4.0 * phi)
    hrr = em4phi * grr - 1.0
    htt = em4phi * gtt_over_r2 - 1.0
    hpp = em4phi * gpp_over_r2sintheta - 1.0

    zeros = jnp.zeros_like(r)

    # Build the state in the same variable order as the original file.
    initial_state = jnp.stack(
        (
            phi, hrr, htt, hpp,
            zeros, zeros, zeros, zeros,
            zeros, zeros, zeros, lapse,
            zeros, v,
        )
    )

    # JAX boundary functions return an updated array, so keep each result.
    initial_state = grid.fill_inner_boundary(initial_state)

    # Set up the metric tensor h_LL.
    h_LL = jnp.zeros((N, SPACEDIM, SPACEDIM))
    h_LL = h_LL.at[:, i_r, i_r].set(hrr)
    h_LL = h_LL.at[:, i_t, i_t].set(htt)
    h_LL = h_LL.at[:, i_p, i_p].set(hpp)

    first_derivative_indices = jnp.array([idx_hrr, idx_htt, idx_hpp])
    dstate_dr = grid.get_first_derivative(initial_state, first_derivative_indices)
    dhrr_dr, dhtt_dr, dhpp_dr = dstate_dr[first_derivative_indices]

    # This is d h_ij / dx^k = dh_dx[x, i, j, k].
    d1_h_dx = jnp.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
    d1_h_dx = d1_h_dx.at[:, i_r, i_r, i_r].set(dhrr_dr)
    d1_h_dx = d1_h_dx.at[:, i_t, i_t, i_r].set(dhtt_dr)
    d1_h_dx = d1_h_dx.at[:, i_p, i_p, i_r].set(dhpp_dr)

    # Unscaled bar_gamma_ij and bar_gamma^ij.
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)

    # The connections Delta^i, Delta^i_jk and Delta_ijk.
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(
        r, h_LL, d1_h_dx, background
    )
    initial_state = initial_state.at[idx_lambdar].set(Delta_U[:, i_r])

    # Fill boundary cells for lambdar.
    initial_state = grid.fill_outer_boundary(
        initial_state, jnp.array([idx_lambdar])
    )
    initial_state = grid.fill_inner_boundary(
        initial_state, jnp.array([idx_lambdar])
    )

    return initial_state.reshape(-1)