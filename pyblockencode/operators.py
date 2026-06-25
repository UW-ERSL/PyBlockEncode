"""
operators.py — Classical reference matrices for pyblockencode.

Each function returns the *trimmed* (Dirichlet interior) stiffness or
mass matrix as a dense numpy array, on an N×N grid with N = 2**m nodes
per dimension.  These are the ground-truth targets that BlockEncoding.verify()
checks against.

Poisson
-------
poisson_1d_fdm(m)        tridiag(-1,2,-1)  of size N×N
poisson_1d_fem(m)        FEM stiffness:  (1/h)*tridiag(-1,2,-1) as interior
poisson_2d_fdm(m)        I⊗T + T⊗I  (isotropic, unit spacing)
poisson_2d_fem(m)        My⊗Kx + Ky⊗Mx  (bilinear Q1)
poisson_3d_fdm(m)        I⊗I⊗T + I⊗T⊗I + T⊗I⊗I
poisson_3d_fem(m)        Mz⊗My⊗Kx + Mz⊗Ky⊗Mx + Kz⊗My⊗Mx  (trilinear Q1)

Elasticity
----------
elasticity_q4(mx, my, E, nu)   2D plane-stress Q4 global stiffness (dense)

All matrices are returned as np.ndarray (dense).  For the block-encoding
harness we work with dense matrices; callers can convert with scipy.sparse
if needed.
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# 1-D primitives
# ---------------------------------------------------------------------------

def _K1(N: int) -> np.ndarray:
    """Interior 1D FEM/FDM stiffness: tridiag(-1,2,-1), size N×N."""
    A = 2.0 * np.eye(N)
    A -= np.diag(np.ones(N - 1), 1)
    A -= np.diag(np.ones(N - 1), -1)
    return A

def _M1(N: int) -> np.ndarray:
    """Interior 1D consistent-mass: tridiag(1,4,1)/6, size N×N."""
    A = 4.0 * np.eye(N)
    A += np.diag(np.ones(N - 1), 1)
    A += np.diag(np.ones(N - 1), -1)
    return A / 6.0

def _G1(N: int) -> np.ndarray:
    """Interior 1D gradient coupling: tridiag(-1,0,1)/2, size N×N."""
    A = np.diag(np.ones(N - 1), 1)
    A -= np.diag(np.ones(N - 1), -1)
    return A / 2.0

# ---------------------------------------------------------------------------
# Poisson — 1D
# ---------------------------------------------------------------------------

def poisson_1d_fdm(m: int) -> np.ndarray:
    """
    1D FDM Poisson interior stiffness on N = 2**m interior points.

    Operator: -u'' = f, uniform h, Dirichlet BCs.
    Returns tridiag(-1, 2, -1) of shape (N, N).
    Same as 1D FEM stiffness (they coincide for linear elements at unit h).
    """
    return _K1(2 ** m)

def poisson_1d_fem(m: int) -> np.ndarray:
    """
    1D FEM Poisson interior stiffness on N = 2**m interior nodes.

    Identical to FDM for linear (P1) elements with uniform mesh:
    K_ij = ∫ φ'_i φ'_j dx  →  tridiag(-1, 2, -1)  (after factoring out 1/h,
    which cancels against h from the load vector).  For the block-encoding
    purposes we use the dimensionless form.
    """
    return _K1(2 ** m)

# ---------------------------------------------------------------------------
# Poisson — 2D
# ---------------------------------------------------------------------------

def poisson_2d_fdm(m: int) -> np.ndarray:
    """
    2D FDM Poisson interior stiffness on N×N interior grid, N = 2**m.

    Five-point Laplacian on a square unit domain:
        K = I_N ⊗ T_N  +  T_N ⊗ I_N
    where T_N = tridiag(-1, 2, -1).  Returns dense array of shape (N², N²).
    """
    N = 2 ** m
    T = _K1(N)
    I = np.eye(N)
    return np.kron(I, T) + np.kron(T, I)

def poisson_2d_fem(m: int) -> np.ndarray:
    """
    2D FEM Poisson interior stiffness on N×N interior grid, N = 2**m.

    Bilinear Q1 elements on a square unit domain:
        K = M_y ⊗ K_x  +  K_y ⊗ M_x
    where K_x = K_y = tridiag(-1,2,-1) and M_x = M_y = tridiag(1,4,1)/6.
    Returns dense array of shape (N², N²).
    """
    N = 2 ** m
    K1 = _K1(N)
    M1 = _M1(N)
    return np.kron(M1, K1) + np.kron(K1, M1)

# ---------------------------------------------------------------------------
# Poisson — 3D
# ---------------------------------------------------------------------------

def poisson_3d_fdm(m: int) -> np.ndarray:
    """
    3D FDM Poisson interior stiffness on N³ interior grid, N = 2**m.

    Seven-point Laplacian:
        K = I⊗I⊗T  +  I⊗T⊗I  +  T⊗I⊗I
    Returns dense array of shape (N³, N³).  Use small m (m≤2 for testing).
    """
    N = 2 ** m
    T = _K1(N)
    I = np.eye(N)
    II = np.kron(I, I)
    IT = np.kron(I, T)
    TI = np.kron(T, I)
    return np.kron(I, IT) + np.kron(I, TI) + np.kron(T, II)

def poisson_3d_fem(m: int) -> np.ndarray:
    """
    3D FEM Poisson interior stiffness on N³ interior grid, N = 2**m.

    Trilinear Q1 elements:
        K = M_z⊗M_y⊗K_x  +  M_z⊗K_y⊗M_x  +  K_z⊗M_y⊗M_x
    Returns dense array of shape (N³, N³).  Use small m (m≤2 for testing).
    """
    N = 2 ** m
    K1 = _K1(N)
    M1 = _M1(N)
    return (np.kron(np.kron(M1, M1), K1)
          + np.kron(np.kron(M1, K1), M1)
          + np.kron(np.kron(K1, M1), M1))

# ---------------------------------------------------------------------------
# Elasticity — 2D plane stress Q4
# ---------------------------------------------------------------------------

def elasticity_q4(mx: int, my: int, E: float = 1.0,
                  nu: float = 0.3) -> np.ndarray:
    """
    2D plane-stress Q4 global stiffness matrix (Dirichlet interior).

    Uses the exact Kronecker decomposition:
        C   = E / (1 - nu²)
        Kxx = C ( K1x⊗M1x + (1-nu)/2 · M1x⊗K1x )   [x varies fastest]
        Kyy = C ( (1-nu)/2 · K1x⊗M1x + M1x⊗K1x )
        Kxy = -C (1+nu)/2 · G1x⊗G1x

    The global matrix uses DOF-innermost ordering to match the block-encoding
    unitary convention: index = (jx*N + jy)*2 + d, with d ∈ {0=x, 1=y}.
    Equivalently:

        K = kron(Kxx, [[1,0],[0,0]])  +  kron(Kxy, [[0,1],[0,0]])
          + kron(Kxy^T, [[0,0],[1,0]]) + kron(Kyy, [[0,0],[0,1]])

    Parameters
    ----------
    mx, my : int
        Number of qubits per dimension; N = 2**m interior nodes per direction.
        For square grids (mx == my), set both to the same value.
    E, nu  : float
        Young's modulus and Poisson's ratio.

    Returns
    -------
    K : np.ndarray of shape (2·Nx·Ny, 2·Nx·Ny)
        DOF ordering: (jx, jy, d) with d innermost (fastest-varying).
    """
    Nx = 2 ** mx
    Ny = 2 ** my
    C = E / (1.0 - nu ** 2)

    K1x, M1x, G1x = _K1(Nx), _M1(Nx), _G1(Nx)
    K1y, M1y, G1y = _K1(Ny), _M1(Ny), _G1(Ny)

    # Spatial blocks — x varies fastest in the kron ordering (y outermost)
    Kxx = C * (np.kron(K1y, M1x) + (1.0 - nu) / 2.0 * np.kron(M1y, K1x))
    Kyy = C * ((1.0 - nu) / 2.0 * np.kron(K1y, M1x) + np.kron(M1y, K1x))
    Kxy = -C * (1.0 + nu) / 2.0 * np.kron(G1y, G1x)

    # DOF-innermost assembly via kron(K_spatial, E_dof)
    Exx = np.array([[1.0, 0.0], [0.0, 0.0]])
    Eyy = np.array([[0.0, 0.0], [0.0, 1.0]])
    Exy = np.array([[0.0, 1.0], [0.0, 0.0]])
    Eyx = np.array([[0.0, 0.0], [1.0, 0.0]])

    return (np.kron(Kxx, Exx) + np.kron(Kxy, Exy)
          + np.kron(Kxy.T, Eyx) + np.kron(Kyy, Eyy))
