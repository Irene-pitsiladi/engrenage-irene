import numpy as np
import jax.numpy as jnp

from bssn.bssnstatevariables import *


class StateVector:
    """Creates the full state vector from the bssn vars and the given matter class."""

    def __init__(self, a_matter):

        # Details for the matter state variables
        self.NUM_VARS = NUM_BSSN_VARS + a_matter.NUM_MATTER_VARS
        self.VARIABLE_NAMES = np.concatenate(
            [BSSN_VARIABLE_NAMES, a_matter.VARIABLE_NAMES]
        )
        self.PARITY = jnp.concatenate(
            [jnp.asarray(BSSN_PARITY), jnp.asarray(a_matter.PARITY)]
        )
        self.ASYMP_POWER = jnp.concatenate(
            [jnp.asarray(BSSN_ASYMP_POWER), jnp.asarray(a_matter.ASYMP_POWER)]
        )
        self.ASYMP_OFFSET = jnp.concatenate(
            [jnp.asarray(BSSN_ASYMP_OFFSET), jnp.asarray(a_matter.ASYMP_OFFSET)]
        )
        self.ALL_INDICES = jnp.arange(self.NUM_VARS, dtype=jnp.int32)

    def print_variable_details(self):
        print(self.VARIABLE_NAMES)