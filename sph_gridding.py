"""
SPH kernel deposition onto uniform grids.

Extracted from python_dnelson/util/sphMap.py (Dylan Nelson) to remove the
dependency on non-pip-installable packages. The numerical kernel and algorithm
are identical to the original Arepo/Gadget cubic spline implementation.

Dependencies: numpy, numba
"""

import numpy as np
import threading
from numba import jit


# ---------------------------------------------------------------------------
# Low-level numba kernels
# ---------------------------------------------------------------------------

@jit(nopython=True, nogil=True, cache=True)
def _nearest(x, box_half, box_size):
    """Periodic wrap distance."""
    if box_size == 0.0:
        return x
    if x > box_half:
        return x - box_size
    elif x < -box_half:
        return x + box_size
    else:
        return x


@jit(nopython=True, nogil=True, cache=True)
def _nearest_pos(x, box_size):
    """Periodic wrap position."""
    if box_size == 0.0:
        return x
    if x > box_size:
        return x - box_size
    elif x < 0:
        return x + box_size
    else:
        return x


@jit(nopython=True, nogil=True, cache=True)
def _kernel_coefficients(ndims):
    """Normalization coefficients for the cubic spline kernel."""
    if ndims == 1:
        C1 = 4.0 / 3
        C2 = 8.0
        C3 = 2.6666666667
    if ndims == 2:
        C1 = 5.0 / 7 * 2.546479089470
        C2 = 5.0 / 7 * 15.278874536822
        C3 = 5.0 / 7 * 5.092958178941
    if ndims == 3:
        C1 = 2.546479089470
        C2 = 15.278874536822
        C3 = 5.092958178941
    return C1, C2, C3


@jit(nopython=True, nogil=True, cache=True)
def _getkernel(hinv, r2, C1, C2, C3):
    """Evaluate the cubic spline kernel."""
    u = np.sqrt(r2) * hinv
    if u < 0.5:
        return C1 + C2 * (u - 1.0) * u * u
    else:
        return C3 * (1.0 - u) * (1.0 - u) * (1.0 - u)


# ---------------------------------------------------------------------------
# 3D grid deposition (the core routine used by Grid_halo.py)
# ---------------------------------------------------------------------------

@jit(nopython=True, nogil=True, cache=True)
def _calc_sph_grid(
    pos, hsml, mass, quant,
    dens_out, quant_out,
    box_size_img, box_size_sim, box_cen,
    ndims, n_pixels,
    norm_vol_dens, max_int_proj, min_int_proj,
):
    """Deposit particles onto a 3D grid using SPH kernel deposition."""
    NumPart = pos.shape[0]

    BoxHalf = np.zeros(3, dtype=np.float32)
    for i in range(3):
        BoxHalf[i] = box_size_sim[i] / 2.0

    COEFF_1, COEFF_2, COEFF_3 = _kernel_coefficients(ndims)

    pixelSizeX = box_size_img[0] / n_pixels[0]
    pixelSizeY = box_size_img[1] / n_pixels[1]
    pixelSizeZ = box_size_img[2] / n_pixels[2]

    minSize = min(pixelSizeX, pixelSizeY, pixelSizeZ)
    hsmlMin = 1.001 * minSize * 0.5
    hsmlMax = minSize * 65.0

    for k in range(NumPart):
        p0 = pos[k, 0]
        p1 = pos[k, 1]
        p2 = pos[k, 2]
        h = hsml[k]
        v = mass[k] if mass.size != 2 else mass[0]
        w = quant[k] if quant.size > 1 else 0.0

        if h < hsmlMin:
            h = hsmlMin
        elif h > hsmlMax:
            h = hsmlMax

        if (
            np.abs(_nearest(p0 - box_cen[0], BoxHalf[0], box_size_sim[0]))
            > 0.5 * box_size_img[0] + h
            or np.abs(_nearest(p1 - box_cen[1], BoxHalf[1], box_size_sim[1]))
            > 0.5 * box_size_img[1] + h
            or np.abs(_nearest(p2 - box_cen[2], BoxHalf[2], box_size_sim[2]))
            > 0.5 * box_size_img[2] + h
        ):
            continue

        pos0 = p0 - (box_cen[0] - 0.5 * box_size_img[0])
        pos1 = p1 - (box_cen[1] - 0.5 * box_size_img[1])
        pos2 = p2 - (box_cen[2] - 0.5 * box_size_img[2])

        h2 = h * h
        hinv = 1.0 / h

        x = (np.floor(pos0 / pixelSizeX) + 0.5) * pixelSizeX
        y = (np.floor(pos1 / pixelSizeY) + 0.5) * pixelSizeY
        z = (np.floor(pos2 / pixelSizeZ) + 0.5) * pixelSizeZ

        nx = int(np.floor(h / pixelSizeX + 1))
        ny = int(np.floor(h / pixelSizeY + 1))
        nz = int(np.floor(h / pixelSizeZ + 1))

        # Single-cell fast path
        if nx * ny * nz == 1:
            i = int(x / pixelSizeX)
            j = int(y / pixelSizeY)
            ki = int(z / pixelSizeZ)
            if i < 0 or i >= n_pixels[0] or j < 0 or j >= n_pixels[1] or ki < 0 or ki >= n_pixels[2]:
                continue
            if min_int_proj:
                if v * w < quant_out[i, j, ki]:
                    dens_out[i, j, ki] = v
                    quant_out[i, j, ki] = v * w
            elif max_int_proj:
                if v * w > quant_out[i, j, ki]:
                    dens_out[i, j, ki] = v
                    quant_out[i, j, ki] = v * w
            else:
                dens_out[i, j, ki] += v
                quant_out[i, j, ki] += v * w
            continue

        # Multi-cell: first pass — compute kernel sum for normalization
        if not min_int_proj and not max_int_proj:
            kSum = 0.0
            v_over_sum = 0.0
            for dx in range(-nx, nx + 1):
                for dy in range(-ny, ny + 1):
                    for dz in range(-nz, nz + 1):
                        xx = x + dx * pixelSizeX - pos0
                        yy = y + dy * pixelSizeY - pos1
                        zz = z + dz * pixelSizeZ - pos2
                        r2 = xx * xx + yy * yy + zz * zz
                        if r2 < h2:
                            kSum += _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)
            if kSum < 1e-10:
                continue
            v_over_sum = v / kSum

        # Multi-cell: second pass — distribute contributions
        for dx in range(-nx, nx + 1):
            for dy in range(-ny, ny + 1):
                for dz in range(-nz, nz + 1):
                    xxx = _nearest_pos(x + dx * pixelSizeX, box_size_sim[0])
                    yyy = _nearest_pos(y + dy * pixelSizeY, box_size_sim[1])
                    zzz = _nearest_pos(z + dz * pixelSizeZ, box_size_sim[2])

                    i = int(xxx / pixelSizeX)
                    j = int(yyy / pixelSizeY)
                    ki = int(zzz / pixelSizeZ)

                    if i < 0 or i >= n_pixels[0] or j < 0 or j >= n_pixels[1] or ki < 0 or ki >= n_pixels[2]:
                        continue

                    xx = x + dx * pixelSizeX - pos0
                    yy = y + dy * pixelSizeY - pos1
                    zz = z + dz * pixelSizeZ - pos2
                    r2 = xx * xx + yy * yy + zz * zz

                    if r2 < h2:
                        kVal = _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)
                        if min_int_proj:
                            if kVal * v * w < quant_out[i, j, ki]:
                                dens_out[i, j, ki] = kVal * v
                                quant_out[i, j, ki] = kVal * v * w
                        elif max_int_proj:
                            if kVal * v * w > quant_out[i, j, ki]:
                                dens_out[i, j, ki] = kVal * v
                                quant_out[i, j, ki] = kVal * v * w
                        else:
                            dens_out[i, j, ki] += kVal * v_over_sum
                            quant_out[i, j, ki] += kVal * v_over_sum * w

    if norm_vol_dens:
        pixelVol = pixelSizeX * pixelSizeY * pixelSizeZ
        dens_out /= pixelVol


# ---------------------------------------------------------------------------
# 2D projection (included for completeness / future use)
# ---------------------------------------------------------------------------

@jit(nopython=True, nogil=True, cache=True)
def _calc_sph_map(
    pos, hsml, mass, quant,
    dens_out, quant_out,
    box_size_img, box_size_sim, box_cen,
    axes, ndims, n_pixels,
    norm_col_dens, max_int_proj, min_int_proj,
):
    """Deposit particles onto a 2D image using SPH kernel projection."""
    NumPart = pos.shape[0]
    axis3 = 3 - axes[0] - axes[1]

    BoxHalf = np.zeros(3, dtype=np.float32)
    for i in range(3):
        BoxHalf[i] = box_size_sim[i] / 2.0

    COEFF_1, COEFF_2, COEFF_3 = _kernel_coefficients(ndims)

    pixelSizeX = box_size_img[0] / n_pixels[0]
    pixelSizeY = box_size_img[1] / n_pixels[1]
    pixelArea = pixelSizeX * pixelSizeY

    if pixelSizeX < pixelSizeY:
        hsmlMin = 1.001 * pixelSizeX * 0.5
        hsmlMax = pixelSizeX * 500.0
    else:
        hsmlMin = 1.001 * pixelSizeY * 0.5
        hsmlMax = pixelSizeY * 500.0

    for k in range(NumPart):
        p0 = pos[k, axes[0]]
        p1 = pos[k, axes[1]]
        p2 = pos[k, axis3] if pos.shape[1] == 3 else 0.0
        h = hsml[k]
        v = mass[k] if mass.size != 2 else mass[0]
        w = quant[k] if quant.size > 1 else 0.0

        if pos.shape[1] == 3:
            if (
                np.abs(_nearest(p2 - box_cen[2], BoxHalf[2], box_size_sim[2]))
                > 0.5 * box_size_img[2] + h
            ):
                continue

        if h < hsmlMin:
            h = hsmlMin
        elif h > hsmlMax:
            h = hsmlMax

        if (
            np.abs(_nearest(p0 - box_cen[0], BoxHalf[0], box_size_sim[0]))
            > 0.5 * box_size_img[0] + h
            or np.abs(_nearest(p1 - box_cen[1], BoxHalf[1], box_size_sim[1]))
            > 0.5 * box_size_img[1] + h
        ):
            continue

        pos0 = p0 - (box_cen[0] - 0.5 * box_size_img[0])
        pos1 = p1 - (box_cen[1] - 0.5 * box_size_img[1])

        h2 = h * h
        hinv = 1.0 / h

        x = (np.floor(pos0 / pixelSizeX) + 0.5) * pixelSizeX
        y = (np.floor(pos1 / pixelSizeY) + 0.5) * pixelSizeY

        nx = int(np.floor(h / pixelSizeX + 1))
        ny = int(np.floor(h / pixelSizeY + 1))

        if nx * ny == 1:
            i = int(x / pixelSizeX)
            j = int(y / pixelSizeY)
            if i < 0 or i >= n_pixels[0] or j < 0 or j >= n_pixels[1]:
                continue
            if min_int_proj:
                if v * w < quant_out[i, j]:
                    dens_out[i, j] = v
                    quant_out[i, j] = v * w
            elif max_int_proj:
                if v * w > quant_out[i, j]:
                    dens_out[i, j] = v
                    quant_out[i, j] = v * w
            else:
                dens_out[i, j] += v
                quant_out[i, j] += v * w
            continue

        if not min_int_proj and not max_int_proj:
            kSum = 0.0
            v_over_sum = 0.0
            for dx in range(-nx, nx + 1):
                for dy in range(-ny, ny + 1):
                    xx = x + dx * pixelSizeX - pos0
                    yy = y + dy * pixelSizeY - pos1
                    r2 = xx * xx + yy * yy
                    if r2 < h2:
                        kSum += _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)
            if kSum < 1e-10:
                continue
            v_over_sum = v / kSum

        for dx in range(-nx, nx + 1):
            for dy in range(-ny, ny + 1):
                xxx = _nearest_pos(x + dx * pixelSizeX, box_size_sim[0])
                yyy = _nearest_pos(y + dy * pixelSizeY, box_size_sim[1])

                i = int(xxx / pixelSizeX)
                j = int(yyy / pixelSizeY)

                if i < 0 or i >= n_pixels[0] or j < 0 or j >= n_pixels[1]:
                    continue

                xx = x + dx * pixelSizeX - pos0
                yy = y + dy * pixelSizeY - pos1
                r2 = xx * xx + yy * yy

                if r2 < h2:
                    kVal = _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)
                    if min_int_proj:
                        if kVal * v * w < quant_out[i, j]:
                            dens_out[i, j] = kVal * v
                            quant_out[i, j] = kVal * v * w
                    elif max_int_proj:
                        if kVal * v * w > quant_out[i, j]:
                            dens_out[i, j] = kVal * v
                            quant_out[i, j] = kVal * v * w
                    else:
                        dens_out[i, j] += kVal * v_over_sum
                        quant_out[i, j] += kVal * v_over_sum * w

    if norm_col_dens:
        dens_out /= pixelArea


# ---------------------------------------------------------------------------
# Threading helper
# ---------------------------------------------------------------------------

def _p_split(array, num_procs, cur_proc):
    """Split an array into num_procs segments and return the cur_proc-th one."""
    if num_procs == 1:
        return array
    split_size = int(np.floor(len(array) / num_procs))
    if cur_proc == num_procs - 1:
        return array[cur_proc * split_size:]
    return array[cur_proc * split_size:(cur_proc + 1) * split_size]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sph_map(
    pos, hsml, mass, quant, axes, box_size_img, box_size_sim, box_cen,
    n_pixels, ndims, col_dens=False, n_threads=1, multi=False,
):
    """
    SPH kernel deposition onto a 2D image or 3D grid.

    Parameters
    ----------
    pos : (N, 3) float32 array
        Particle positions.
    hsml : (N,) float32 array
        Smoothing lengths.
    mass : (N,) float32 array
        Particle masses.
    quant : (N,) float32 array or None
        Quantity to compute mass-weighted grid of. None to skip.
    axes : (2,) list
        Projection axes, e.g. [0, 1] for x-y.
    box_size_img : (2,) or (3,) array
        Physical size of the output image/grid.
    box_size_sim : (3,) array
        Simulation box size for periodic wrapping (0 = non-periodic).
    box_cen : (3,) array
        Center of the output image/grid.
    n_pixels : (2,) or (3,) array of int
        Number of pixels/cells in each dimension.
    ndims : int
        Dimensionality of the simulation (1, 2, or 3).
    col_dens : bool
        If True, normalize by pixel area/volume.
    n_threads : int
        Number of threads.
    multi : bool
        If True, return (dens, quant) tuple without normalizing quant by dens.

    Returns
    -------
    If quant is None: mass/density grid.
    If quant is given and multi=False: mass-weighted quantity grid.
    If quant is given and multi=True: (mass grid, mass*quant grid) tuple.
    """
    # Input validation and type coercion
    if not isinstance(box_size_sim, (float, int)):
        assert len(box_size_sim) == 3
    else:
        box_size_sim = [box_size_sim, box_size_sim, box_size_sim]

    if hsml.dtype == np.float64:
        hsml = hsml.astype("float32")
    if mass.dtype == np.float64:
        mass = mass.astype("float32")

    if quant is None:
        quant = np.array([0], dtype="float32")
    elif quant.dtype != np.float32:
        quant = quant.astype("float32")

    if mass.size == 1:
        mass = np.array([mass.item(), mass.item()], dtype="float32")

    box_size_img = np.asarray(box_size_img, dtype=np.float64)
    box_size_sim = np.asarray(box_size_sim, dtype=np.float64)
    box_cen = np.asarray(box_cen, dtype=np.float64)
    n_pixels = np.asarray(n_pixels, dtype=np.int64)
    axes = np.asarray(axes, dtype=np.int64)

    is_3d = len(n_pixels) == 3

    # Allocate output
    rDens = np.zeros(n_pixels, dtype="float32")
    rQuant = np.zeros(n_pixels, dtype="float32")

    if n_threads == 1:
        if is_3d:
            _calc_sph_grid(
                pos, hsml, mass, quant, rDens, rQuant,
                box_size_img, box_size_sim, box_cen,
                ndims, n_pixels, col_dens, False, False,
            )
        else:
            _calc_sph_map(
                pos, hsml, mass, quant, rDens, rQuant,
                box_size_img, box_size_sim, box_cen,
                axes, ndims, n_pixels, col_dens, False, False,
            )
    else:
        # Multithreaded
        class MapThread(threading.Thread):
            def __init__(self, thread_num, total_threads):
                super().__init__()
                self.rDens = np.zeros(n_pixels, dtype="float32")
                self.rQuant = np.zeros(n_pixels, dtype="float32")

                self.pos = _p_split(pos, total_threads, thread_num)
                self.hsml = _p_split(hsml, total_threads, thread_num)
                self.mass = (
                    _p_split(mass, total_threads, thread_num)
                    if mass.size != 2 else mass
                )
                self.quant = (
                    _p_split(quant, total_threads, thread_num)
                    if quant.size > 1 else quant
                )

            def run(self):
                if is_3d:
                    _calc_sph_grid(
                        self.pos, self.hsml, self.mass, self.quant,
                        self.rDens, self.rQuant,
                        box_size_img, box_size_sim, box_cen,
                        ndims, n_pixels, col_dens, False, False,
                    )
                else:
                    _calc_sph_map(
                        self.pos, self.hsml, self.mass, self.quant,
                        self.rDens, self.rQuant,
                        box_size_img, box_size_sim, box_cen,
                        axes, ndims, n_pixels, col_dens, False, False,
                    )

        threads = [MapThread(i, n_threads) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            rQuant += t.rQuant
            rDens += t.rDens

    # Return
    if multi:
        return (rDens.T, rQuant.T) if not is_3d else (rDens, rQuant)

    if quant.size > 1:
        w = np.where(rDens > 0.0)
        rQuant[w] /= rDens[w]
        return rQuant.T if not is_3d else rQuant

    return rDens.T if not is_3d else rDens


def deposit_particles_on_grid(
    gas_parts, method, quants, box_size_parts, grid_shape, grid_size,
    grid_cen, n_threads, mass_key="Masses",
):
    """
    Drop-in replacement for pyTNG.gridding.depositParticlesOnGrid.

    Deposits particle masses and associated quantities on a 3D grid using
    SPH kernel deposition.

    Parameters
    ----------
    gas_parts : dict
        Particle dictionary with 'Coordinates', 'hsml', and mass_key arrays.
    method : str
        'sphKernelDep' (only supported method).
    quants : list of str or None
        Keys in gas_parts for quantities to deposit (mass-weighted).
    box_size_parts : (3,) array
        Simulation box size for periodic wrapping (0 = non-periodic).
    grid_shape : (3,) array of int
        Number of cells in each dimension.
    grid_size : (3,) array
        Physical size of the grid.
    grid_cen : (3,) array
        Center of the grid.
    n_threads : int
        Number of threads.
    mass_key : str
        Key for particle masses in gas_parts.

    Returns
    -------
    dict : Grids for mass_key and each quantity in quants.
    """
    if method != "sphKernelDep":
        raise ValueError(f"Unknown gridding method: {method}. Only 'sphKernelDep' is supported.")

    if "hsml" not in gas_parts:
        raise KeyError("'hsml' not found in input dict, required for sphKernelDep.")

    if quants:
        if any(x not in gas_parts for x in quants):
            raise ValueError("Got unknown quantity to grid.")

    r = {}
    if quants:
        rDens, qGrids = sph_map(
            gas_parts["Coordinates"], gas_parts["hsml"], gas_parts[mass_key],
            None, [0, 1], grid_size, box_size_parts, grid_cen,
            grid_shape, 3, n_threads=n_threads, multi=True,
        )
        # Now compute mass-weighted quantity grids
        r[mass_key] = rDens
        for key in quants:
            qgrid = sph_map(
                gas_parts["Coordinates"], gas_parts["hsml"], gas_parts[mass_key],
                gas_parts[key], [0, 1], grid_size, box_size_parts, grid_cen,
                grid_shape, 3, n_threads=n_threads, multi=True,
            )
            r[key] = qgrid[1]  # mass * quantity grid
    else:
        rDens = sph_map(
            gas_parts["Coordinates"], gas_parts["hsml"], gas_parts[mass_key],
            None, [0, 1], grid_size, box_size_parts, grid_cen,
            grid_shape, 3, n_threads=n_threads,
        )
        r[mass_key] = rDens

    return r
