import jax.numpy as jnp

from core.spacing import Spacing, NUM_GHOSTS

class Derivatives:
    """
    Represents the derivatives on a grid defines by the points in spacing.
    It uses finite difference method: the code provide derivation matrices up to order 6.
    They may be used to compute derivatives of functions using dot product.
    On the ghost points, (3 assumed by default), the derivatives are not computed.
    """
    def __init__(
        self,
        spacing: Spacing,
    ):

        self.dnr_dxn = spacing.dnr_dxn
        self.dx = spacing.dx
        self.num_points = n = self.dnr_dxn.shape[1]

        # Create a matrix with stencils for derivative from order 0 to order 6
        self.dxn_matrix = jnp.zeros((7, n, n))
        self.drn_matrix = jnp.zeros((7, n, n))
        self.compute_dxn_matrix()
        self.compute_drn_matrix()

        # Compute left and right advection matrices
        self.advec_x_matrix = jnp.zeros((2, n, n))
        self.compute_advec_x_matrix()

    def compute_dxn_matrix(self):
        """
        Fill the derivation matrices for x coordinate, i.e. evenly spaced points.
        The spacing is assumed to be one, scaling is done afterward.
        The finite difference coefficients may be found on
        https://en.wikipedia.org/wiki/Finite_difference_coefficient.
        """
        # For readability
        n = self.num_points
        # 0. Oth order derivative is identity (not used but for completeness).
        self.dxn_matrix=self.dxn_matrix.at[0].set(jnp.eye(n))
        # 1. Create fourth order first derivation matrix.
        # Stencil [1/12, -2/3, 0, 2/3, -1/12]
        self.dxn_matrix=self.dxn_matrix.at[1].set(
        
            1 / 12 * jnp.eye(n, k=-2)
            - 2 / 3 * jnp.eye(n, k=-1)
            + 2 / 3 * jnp.eye(n, k=1)
            - 1 / 12 * jnp.eye(n, k=2)
        )
        # 2. Create fourth order second derivation matrix.
        # Stencil [-1/12, 4/3, -5/2, 4/3, -1/12]
        self.dxn_matrix=self.dxn_matrix.at[2].set(
            -1 / 12 * jnp.eye(n, k=-2)
            + 4 / 3 * jnp.eye(n, k=-1)
            - 5 / 2 * jnp.eye(n, k=-0)
            + 4 / 3 * jnp.eye(n, k=1)
            - 1 / 12 * jnp.eye(n, k=2)
        )
        # 3. Create second order third derivation matrix.
        # Stencil [-1/2, 1, 0, -1, 1/2]
        self.dxn_matrix=self.dxn_matrix.at[3].set(
            -1 / 2 * jnp.eye(n, k=-2)
            + 1 * jnp.eye(n, k=-1)
            - 1 * jnp.eye(n, k=1)
            + 1 / 2 * jnp.eye(n, k=2)
        )
        # 4. Create second order fourth derivation matrix.
        # Stencil: [-1/2, 1, 0, -1, 1/2]
        self.dxn_matrix=self.dxn_matrix.at[4].set(
            1 * jnp.eye(n, k=-2)
            - 4 * jnp.eye(n, k=-1)
            + 6 * jnp.eye(n, k=0)
            - 4 * jnp.eye(n, k=1)
            + 1 * jnp.eye(n, k=2)
        )
        # 5. Create second order fifth derivation matrix.
        # Stencil: [-1, 2, -5/2, 0, 5/2, -2, 1]
        self.dxn_matrix=self.dxn_matrix.at[5].set(
            -1 / 2 * jnp.eye(n, k=-3)
            + 2 * jnp.eye(n, k=-2)
            - 5 / 2 * jnp.eye(n, k=-1)
            + 5 / 2 * jnp.eye(n, k=1)
            - 2 * jnp.eye(n, k=2)
            + 1 / 2 * jnp.eye(n, k=3)
        )
        # 6. Create second order sixth derivation matrix.
        # Stencil: [1, -6, 15, -20, 15, -6, 1]
        self.dxn_matrix=self.dxn_matrix.at[6].set(
            1 * jnp.eye(n, k=-3)
            - 6 * jnp.eye(n, k=-2)
            + 15 * jnp.eye(n, k=-1)
            - 20 * jnp.eye(n, k=0)
            + 15 * jnp.eye(n, k=1)
            - 6 * jnp.eye(n, k=2)
            + 1 * jnp.eye(n, k=3)
        )

        # Do not compute derivatives for ghost points.
       
        self.dxn_matrix=self.dxn_matrix.at[:, :NUM_GHOSTS].set(0)
        self.dxn_matrix=self.dxn_matrix.at[:, -NUM_GHOSTS:].set(0)

    def compute_drn_matrix(self):
        # Quantities need to be scaled for real derivatives (by a factor of 1 / (dr_dx * h)**n for nth derivative)
        dr_dx = self.dnr_dxn[1]
        self.drn_matrix=self.drn_matrix.at[0].set(self.dxn_matrix[0])
        self.drn_matrix=self.drn_matrix.at[1].set(self.dxn_matrix[1])
        
        self.drn_matrix=self.drn_matrix.at[2].set(
            self.dxn_matrix[2]
            - self.dx * jnp.diag(self.dnr_dxn[2] / dr_dx) @ self.drn_matrix[1]
        )
        
        
        self.drn_matrix=self.drn_matrix.at[3].set(
                self.dxn_matrix[3]
                - self.dx * jnp.diag(3 * self.dnr_dxn[2] / dr_dx) @ self.drn_matrix[2]
                - self.dx ** 2 * jnp.diag(self.dnr_dxn[3] / dr_dx) @ self.drn_matrix[1]
        )
        self.drn_matrix=self.drn_matrix.at[4].set(
                self.dxn_matrix[4]
                - self.dx * jnp.diag(6 * self.dnr_dxn[2] / dr_dx) @ self.drn_matrix[3]
                - self.dx ** 2 * jnp.diag(4 * self.dnr_dxn[3] / dr_dx + 3 * (self.dnr_dxn[2] / dr_dx) ** 2) @ self.drn_matrix[2]
                - self.dx ** 3 * jnp.diag(self.dnr_dxn[4] / dr_dx) @ self.drn_matrix[1]
        )
        self.drn_matrix=self.drn_matrix.at[5].set(
                self.dxn_matrix[5]
                - self.dx * jnp.diag(10 * self.dnr_dxn[2] / dr_dx) @ self.drn_matrix[4]
                - self.dx ** 2 * jnp.diag(10 * self.dnr_dxn[3] / dr_dx + 15 * (self.dnr_dxn[2] / dr_dx) ** 2) @ self.drn_matrix[3]
                - self.dx ** 3 * jnp.diag(10 * self.dnr_dxn[3] * self.dnr_dxn[2] / dr_dx ** 2 + 5 * self.dnr_dxn[4] / dr_dx) @ self.drn_matrix[2]
                - self.dx ** 4 * jnp.diag(self.dnr_dxn[5] / dr_dx) @ self.drn_matrix[1]
        )
        self.drn_matrix=self.drn_matrix.at[6].set(
                self.dxn_matrix[6]
                - self.dx * jnp.diag(15 * self.dnr_dxn[2] / dr_dx) @ self.drn_matrix[5]
                - self.dx ** 2 * jnp.diag(45 * (self.dnr_dxn[2] / dr_dx) ** 2 + 20 * self.dnr_dxn[3] / dr_dx) @ self.drn_matrix[4]
                - self.dx ** 3 * jnp.diag(15 * self.dnr_dxn[4] / dr_dx + 60 * self.dnr_dxn[3] * self.dnr_dxn[2] / dr_dx ** 2 + 15 * (self.dnr_dxn[2] / dr_dx) ** 3) @ self.drn_matrix[3]
                - self.dx ** 4 * jnp.diag(6 * self.dnr_dxn[5] / dr_dx + 15 * self.dnr_dxn[4] * self.dnr_dxn[2] / dr_dx ** 2 + 10 * (self.dnr_dxn[3] / dr_dx) ** 2) @ self.drn_matrix[2]
                - self.dx ** 5 * jnp.diag(self.dnr_dxn[6] / dr_dx) @ self.drn_matrix[1]
        )

    def compute_advec_x_matrix(self):
        # Left advection.
        # Third order backward stencil: [-1/3, 3/2, -3, 11/6]
        self.advec_x_matrix =self.advec_x_matrix.at[0].set(
                -1 / 3 * jnp.eye(self.num_points, k=-3)
                + 3 / 2 * jnp.eye(self.num_points, k=-2)
                - 3 * jnp.eye(self.num_points, k=-1)
                + 11 / 6 * jnp.eye(self.num_points, k=0)
        )
        # Right advection.
        # Third order forward stencil: [-11/6, 3, -3/2, 1/3]
        self.advec_x_matrix=self.advec_x_matrix.at[1].set(
            -11 / 6 * jnp.eye(self.num_points, k=0)
            + 3 * jnp.eye(self.num_points, k=1)
            - 3 / 2 * jnp.eye(self.num_points, k=2)
            + 1 / 3 * jnp.eye(self.num_points, k=3)
        )

        # Do not compute advection where it is not possible because of boundaries.
        self.advec_x_matrix=self.advec_x_matrix.at[0, :NUM_GHOSTS].set(0)
        self.advec_x_matrix=self.advec_x_matrix.at[1, -NUM_GHOSTS:].set(0)
