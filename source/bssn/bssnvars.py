import jax.numpy as jnp

from core.statevector import *
from backgrounds.sphericalbackground import *


# These classes just allow us to hold all the metric and derivative quantities
# in a nice packaged form and map the non trivial components into full vector
# and tensor forms

class BSSNVars:
    """
    Holder for the BSSN quantities in tensor form
    """
    def __init__(self, N):

        self.N = N
        self.h_LL = jnp.zeros([N, SPACEDIM, SPACEDIM])
        self.a_LL = jnp.zeros([N, SPACEDIM, SPACEDIM])
        self.lambda_U = jnp.zeros([N, SPACEDIM])
        self.shift_U = jnp.zeros([N, SPACEDIM])
        self.b_U = jnp.zeros([N, SPACEDIM])
        self.lapse = jnp.zeros([N])
        self.phi = jnp.zeros([N])
        self.K = jnp.zeros([N])

    def set_bssn_vars(self, state):
        """
        Populate the bssn vars from the full state
        """

        # Unpack variables from current_state - see statevector.py
        # Assumes BSSN vars are first, which they should be
        (phi, hrr, htt, hpp, K, arr, att, app, lambdar,
         shiftr, br, lapse) = state[0:NUM_BSSN_VARS]

        self.h_LL = self.h_LL.at[:, i_r, i_r].set(hrr)
        self.h_LL = self.h_LL.at[:, i_t, i_t].set(htt)
        self.h_LL = self.h_LL.at[:, i_p, i_p].set(hpp)

        self.a_LL = self.a_LL.at[:, i_r, i_r].set(arr)
        self.a_LL = self.a_LL.at[:, i_t, i_t].set(att)
        self.a_LL = self.a_LL.at[:, i_p, i_p].set(app)

        self.lambda_U = self.lambda_U.at[:, i_r].set(lambdar)
        self.shift_U = self.shift_U.at[:, i_r].set(shiftr)
        self.b_U = self.b_U.at[:, i_r].set(br)

        self.lapse = self.lapse.at[:].set(lapse)
        self.phi = self.phi.at[:].set(phi)
        self.K = self.K.at[:].set(K)

    def set_bssn_state_vars(self):
        """
        Populate the state vector from the bssn vars values
        """

        bssn_state = jnp.zeros(self.N * NUM_BSSN_VARS)
        bssn_state = bssn_state.reshape(NUM_BSSN_VARS, -1)

        # Assign parts of the state vector to the different vars
        bssn_state = bssn_state.at[0].set(self.phi)
        bssn_state = bssn_state.at[1].set(self.h_LL[:, i_r, i_r])
        bssn_state = bssn_state.at[2].set(self.h_LL[:, i_t, i_t])
        bssn_state = bssn_state.at[3].set(self.h_LL[:, i_p, i_p])
        bssn_state = bssn_state.at[4].set(self.K)
        bssn_state = bssn_state.at[5].set(self.a_LL[:, i_r, i_r])
        bssn_state = bssn_state.at[6].set(self.a_LL[:, i_t, i_t])
        bssn_state = bssn_state.at[7].set(self.a_LL[:, i_p, i_p])
        bssn_state = bssn_state.at[8].set(self.lambda_U[:, i_r])
        bssn_state = bssn_state.at[9].set(self.shift_U[:, i_r])
        bssn_state = bssn_state.at[10].set(self.b_U[:, i_r])
        bssn_state = bssn_state.at[11].set(self.lapse)

        return bssn_state


class BSSNFirstDerivs:
    """
    Holder for the BSSN d1 quantities in tensor form, last indices are the
    derivative ones by convention
    """
    def __init__(self, N):

        self.first_derivative_indices = jnp.array([
            idx_phi, idx_hrr, idx_htt, idx_hpp, idx_K, idx_arr, idx_att,
            idx_app, idx_lambdar, idx_shiftr, idx_lapse
        ])

        self.h_LL = jnp.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
        self.a_LL = jnp.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
        self.shift_U = jnp.zeros([N, SPACEDIM, SPACEDIM])
        self.lambda_U = jnp.zeros([N, SPACEDIM, SPACEDIM])
        self.phi = jnp.zeros([N, SPACEDIM])
        self.K = jnp.zeros([N, SPACEDIM])
        self.lapse = jnp.zeros([N, SPACEDIM])

    def set_bssn_first_derivs(self, dstate_dr):
        """
        Populate the bssn vars from the variable derivs
        """

        (
            dphi_dr,
            dhrr_dr,
            dhtt_dr,
            dhpp_dr,
            dK_dr,
            darr_dr,
            datt_dr,
            dapp_dr,
            dlambdar_dr,
            dshiftr_dr,
            dlapse_dr,
        ) = dstate_dr[self.first_derivative_indices]

        self.h_LL = self.h_LL.at[:, i_r, i_r, i_r].set(dhrr_dr)
        self.h_LL = self.h_LL.at[:, i_t, i_t, i_r].set(dhtt_dr)
        self.h_LL = self.h_LL.at[:, i_p, i_p, i_r].set(dhpp_dr)

        self.a_LL = self.a_LL.at[:, i_r, i_r, i_r].set(darr_dr)
        self.a_LL = self.a_LL.at[:, i_t, i_t, i_r].set(datt_dr)
        self.a_LL = self.a_LL.at[:, i_p, i_p, i_r].set(dapp_dr)

        self.shift_U = self.shift_U.at[:, i_r, i_r].set(dshiftr_dr)
        self.lambda_U = self.lambda_U.at[:, i_r, i_r].set(dlambdar_dr)

        self.phi = self.phi.at[:, i_r].set(dphi_dr)
        self.K = self.K.at[:, i_r].set(dK_dr)
        self.lapse = self.lapse.at[:, i_r].set(dlapse_dr)


class BSSNSecondDerivs:
    """
    Holder for the BSSN d2 quantities in tensor form, last indices are the
    derivative ones by convention
    """
    def __init__(self, N):

        self.second_derivative_indices = jnp.array([
            idx_phi, idx_hrr, idx_htt, idx_hpp, idx_shiftr, idx_lapse
        ])

        self.h_LL = jnp.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM, SPACEDIM])
        self.shift_U = jnp.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
        self.phi = jnp.zeros([N, SPACEDIM, SPACEDIM])
        self.lapse = jnp.zeros([N, SPACEDIM, SPACEDIM])

    def set_bssn_second_derivs(self, d2state_dr2):
        """
        Populate the bssn vars from the variable derivs
        """

        (
            d2phi_dr2,
            d2hrr_dr2,
            d2htt_dr2,
            d2hpp_dr2,
            d2shiftr_dr2,
            d2lapse_dr2,
        ) = d2state_dr2[self.second_derivative_indices]

        self.h_LL = self.h_LL.at[:, i_r, i_r, i_r, i_r].set(d2hrr_dr2)
        self.h_LL = self.h_LL.at[:, i_t, i_t, i_r, i_r].set(d2htt_dr2)
        self.h_LL = self.h_LL.at[:, i_p, i_p, i_r, i_r].set(d2hpp_dr2)

        self.shift_U = self.shift_U.at[:, i_r, i_r, i_r].set(d2shiftr_dr2)

        self.phi = self.phi.at[:, i_r, i_r].set(d2phi_dr2)
        self.lapse = self.lapse.at[:, i_r, i_r].set(d2lapse_dr2)


class BSSNAdvecDerivs:
    """
    Holder for the BSSN d1 advec quantities in tensor form, last indices are
    the derivative ones by convention
    """
    def __init__(self, N):

        self.advec_indices = jnp.array([
            idx_phi, idx_hrr, idx_htt, idx_hpp, idx_arr, idx_att, idx_app,
            idx_K, idx_lambdar, idx_lapse
        ])

        self.h_LL = jnp.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
        self.a_LL = jnp.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
        self.lambda_U = jnp.zeros([N, SPACEDIM, SPACEDIM])
        self.shift_U = jnp.zeros([N, SPACEDIM, SPACEDIM])
        self.phi = jnp.zeros([N, SPACEDIM])
        self.K = jnp.zeros([N, SPACEDIM])
        self.lapse = jnp.zeros([N, SPACEDIM])

        # temp
        self.u = jnp.zeros([N, SPACEDIM])

    def set_bssn_advec_derivs(self, dstate_dr_advec):
        """
        Populate the bssn vars from the variable derivs
        """

        (
            dphi_dr_advec,
            dhrr_dr_advec,
            dhtt_dr_advec,
            dhpp_dr_advec,
            darr_dr_advec,
            datt_dr_advec,
            dapp_dr_advec,
            dK_dr_advec,
            dlambdar_dr_advec,
            dlapse_dr_advec,
        ) = dstate_dr_advec[self.advec_indices]

        self.h_LL = self.h_LL.at[:, i_r, i_r, i_r].set(dhrr_dr_advec)
        self.h_LL = self.h_LL.at[:, i_t, i_t, i_r].set(dhtt_dr_advec)
        self.h_LL = self.h_LL.at[:, i_p, i_p, i_r].set(dhpp_dr_advec)

        self.a_LL = self.a_LL.at[:, i_r, i_r, i_r].set(darr_dr_advec)
        self.a_LL = self.a_LL.at[:, i_t, i_t, i_r].set(datt_dr_advec)
        self.a_LL = self.a_LL.at[:, i_p, i_p, i_r].set(dapp_dr_advec)

        self.lambda_U = self.lambda_U.at[:, i_r, i_r].set(dlambdar_dr_advec)

        self.phi = self.phi.at[:, i_r].set(dphi_dr_advec)
        self.K = self.K.at[:, i_r].set(dK_dr_advec)
        self.lapse = self.lapse.at[:, i_r].set(dlapse_dr_advec)