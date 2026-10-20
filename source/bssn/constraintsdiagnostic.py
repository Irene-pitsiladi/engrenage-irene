# constraintsdiagnostic.py

import jax.numpy as jnp

from core.grid import *
from bssn.tensoralgebra import *


# The diagnostic function returns the Hamiltonian constraint over the grid.
# It takes the state vector at each time step and returns the spatial profiles
# Ham(r) and Mom(r) at each time step.
def get_constraints_diagnostic(states_over_time, t, grid: Grid, background, matter):

    # For readability
    r = grid.r
    N = grid.num_points

    num_times = states_over_time.size // (grid.NUM_VARS * N)
    Ham = jnp.zeros((num_times, N))
    Mom = jnp.zeros((num_times, N, SPACEDIM))

    # Unpack the vectors at each time
    for i in range(num_times):

        if num_times == 1:
            state = states_over_time
            t_i = t
        else:
            state = states_over_time[i]
            t_i = t[i]

        state = state.reshape(grid.NUM_VARS, -1)

        # Assign the variables to parts of the solution
        N = grid.N
        bssn_vars = BSSNVars(N)
        bssn_vars.set_bssn_vars(state)
        matter.set_matter_vars(state, bssn_vars, grid)

        # Get the derivatives of the BSSN variables in tensor form.
        d1 = grid.get_d1_metric_quantities(state)
        d2 = grid.get_d2_metric_quantities(state)

        # Calculate useful quantities
        em4phi = jnp.exp(-4.0 * bssn_vars.phi)

        # Unscaled bar_gamma_ij and bar_gamma^ij
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(
            r, bssn_vars.h_LL, d1.h_LL, background
        )
        bar_chris = get_bar_christoffel(r, Delta_ULL, background)

        bar_Rij = get_bar_ricci_tensor(
            r,
            bssn_vars.h_LL,
            d1.h_LL,
            d2.h_LL,
            bssn_vars.lambda_U,
            d1.lambda_U,
            Delta_U,
            Delta_ULL,
            Delta_LLL,
            bar_gamma_UU,
            bar_gamma_LL,
            background,
        )
        bar_R = get_trace(bar_Rij, bar_gamma_UU)

        # bar_A^ij and bar_A_ij
        A_UU = get_bar_A_UU(r, bssn_vars, background)
        A_LL = get_bar_A_LL(r, bssn_vars, background)

        # Asquared = bar_A_ij bar_A^ij
        Asquared = get_bar_A_squared(r, bssn_vars, background)

        # Rescaled quantities needed for the momentum constraint
        s_times_d1_a = (
            background.scaling_matrix[:, :, :, jnp.newaxis] * d1.a_LL
        )
        a_times_d1_s = (
            bssn_vars.a_LL[:, :, :, jnp.newaxis]
            * background.d1_scaling_matrix
        )

        # Matter sources
        my_emtensor = matter.get_emtensor(r, bssn_vars, background)

        # Hamiltonian constraint
        ham_i = (
            two_thirds * bssn_vars.K * bssn_vars.K
            - Asquared
            + em4phi
            * (
                bar_R
                - 8.0
                * jnp.einsum(
                    "xij,xi,xj->x",
                    bar_gamma_UU,
                    d1.phi,
                    d1.phi,
                )
                - 8.0
                * jnp.einsum(
                    "xij,xij->x",
                    bar_gamma_UU,
                    d2.phi,
                )
                + 8.0
                * jnp.einsum(
                    "xij,xkij,xk->x",
                    bar_gamma_UU,
                    bar_chris,
                    d1.phi,
                )
            )
            - 2.0 * eight_pi_G * my_emtensor.rho
        )
        Ham = Ham.at[i, :].set(ham_i)

        # Momentum constraint
        mom_i = em4phi[:, jnp.newaxis] * (
            jnp.einsum(
                "xil,xjm,xlmj->xi",
                bar_gamma_UU,
                bar_gamma_UU,
                s_times_d1_a,
            )
            + jnp.einsum(
                "xil,xjm,xlmj->xi",
                bar_gamma_UU,
                bar_gamma_UU,
                a_times_d1_s,
            )
            - jnp.einsum(
                "xil,xjm,xnjl,xnm->xi",
                bar_gamma_UU,
                bar_gamma_UU,
                bar_chris,
                A_LL,
            )
            - jnp.einsum(
                "xil,xjm,xnjm,xln->xi",
                bar_gamma_UU,
                bar_gamma_UU,
                bar_chris,
                A_LL,
            )
            + 6.0
            * jnp.einsum("xij,xj->xi", A_UU, d1.phi)
            - two_thirds
            * jnp.einsum("xij,xj->xi", bar_gamma_UU, d1.K)
            - eight_pi_G
            * jnp.einsum("xij,xj->xi", bar_gamma_UU, my_emtensor.Si)
        )
        Mom = Mom.at[i, :, :].set(mom_i)

        # Fill the endpoints. Each boundary function returns an updated array.
        ham_i = grid.fill_inner_boundary_single_variable(ham_i)
        Ham = Ham.at[i, :].set(ham_i)

        mom_r = grid.fill_inner_boundary_single_variable(Mom[i, :, i_r])
        Mom = Mom.at[i, :, i_r].set(mom_r)

        mom_t = grid.fill_inner_boundary_single_variable(Mom[i, :, i_t])
        Mom = Mom.at[i, :, i_t].set(mom_t)

        mom_p = grid.fill_inner_boundary_single_variable(Mom[i, :, i_p])
        Mom = Mom.at[i, :, i_p].set(mom_p)

    return Ham, Mom