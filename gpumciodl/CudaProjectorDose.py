# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 00:21:27 2016

@author: JonasAdler
"""

# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

import odl
import GPUMCIPy
from gpumci import util

__all__ = ('CudaProjectorDoseSpectrum', )

class CudaProjectorDoseSpectrum(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volume, nVoxels, geometry,
                 energy_min, energy_max, spectrum, material_names):

        self.geometries = util.getProjectionGeometries(geometry)

        # Discretize projection space
        projectionDisc = odl.uniform_discr_frompartition(geometry.det_partition,
                                                         impl='cuda',
                                                         dtype='float32',
                                                         order='F')

        # Discretize the reconstruction space
        reconDisc = odl.uniform_discr_fromintv(volume,
                                               nVoxels,
                                               impl='cuda',
                                               dtype='float32',
                                               order='F')
        reconMatDisc = odl.uniform_discr_fromintv(volume,
                                                  nVoxels,
                                                  impl='cuda',
                                                  dtype='uint8',
                                                  order='F')

        # Extract parameters
        volumeOrigin = volume.min()
        voxelSize = reconDisc.cell_sides
        nPixels = projectionDisc.shape
        domain = odl.ProductSpace(reconDisc, reconMatDisc)
        range = reconDisc

        # Set locals
        self.spectrum = [projectionDisc.element(g.copy()) for g in spectrum]
        self.energy_min = np.asarray(energy_min)
        self.energy_max = np.asarray(energy_max)

        materials = util.getMaterialAttenuationTables(material_names)
        rayleighTables = util.getInteractionTables('rayleigh', material_names)
        comptonTables = util.getInteractionTables('compton', material_names)

        self.forward = GPUMCIPy.GainDoseMC(nVoxels, volumeOrigin, voxelSize,
                                           nPixels, materials, rayleighTables,
                                           comptonTables)

        super().__init__(domain, range)

    def _call(self, data, out):
        # Create projector
        self.forward.setData(data[0].ntuple.data_ptr, data[1].ntuple.data_ptr)

        out.set_zero()

        # Project all geometries
        for geo in self.geometries:
            for i, spec in enumerate(self.spectrum):

                self.forward.project(geo.sourcePosition,
                                     geo.detectorOrigin,
                                     geo.pixelDirectionU,
                                     geo.pixelDirectionV,
                                     out.ntuple.data_ptr,
                                     spec.ntuple.data_ptr,
                                     self.energy_min[i],
                                     self.energy_max[i])


def _get_spectrum(n):
    energies = np.linspace(0.03, 0.09, n, endpoint=False)
    # polynomial with zero at E=0 and E=0.09, peak at E=0.06
    spectrum = 8.33333333e+02 * energies**2 - 9.25925926e+03 * energies**3
    spectrum /= spectrum.sum()
    return energies, spectrum


if __name__ == '__main__':
    """Compute CTDI dose in the elekta icon setup."""

    # Here we can set what type of dose to compute.
    # 'ctdi_center' is the dose in the center.
    # 'ctdi_peripheral' is the average dose in the edges.
    # 'ctdi_w' is the "legal" dose.
    # 'mean' is the mean across the whole phantom.
    dose_type = 'mean'

    # Z-offset describes how far "up" the phantom is,
    # else we assume it is centered in x and y
    z_offset = 10

    # Set geometry parameters
    volumeSize = np.array([224.0, 224.0, 224.0])
    volumeOrigin = np.array([-112.0, -112.0, 0.0])

    detectorSize = np.array([287.04, 264.94])
    detectorOrigin = np.array([-143.52, 0.0])

    sourceAxisDistance = 790.0
    detectorAxisDistance = 210.0

    # Discretization parameters
    nVoxels, nPixels = np.array([448, 448, 448]), np.array([780, 720])
    nProjection = 20

    # Continuous volume
    volume = odl.IntervalProd(volumeOrigin, volumeOrigin+volumeSize)

    # Geometryg
    agrid = odl.uniform_partition(0, 1.2 * np.pi, nProjection)
    dgrid = odl.uniform_partition(detectorOrigin, detectorOrigin+detectorSize,
                                  nPixels)
    geom = odl.tomo.CircularConeFlatGeometry(agrid, dgrid,
                                             src_radius=sourceAxisDistance,
                                             det_radius=detectorAxisDistance)

    energies, spectrum = _get_spectrum(10)
    photons_per_pixel = 100
    spectrum *= photons_per_pixel
    gain = np.ones(nPixels)

    spectrum = np.tile(spectrum[:, None, None], [1, nPixels[0], nPixels[1]])

    materials = ['air', 'water']

    projector = CudaProjectorDoseSpectrum(volume, nVoxels, geom,
                                          energies, energies, spectrum,
                                          materials)

    # Create cylinder
    diam = 80.0 / 112.0  # relative size of ellipse
    ctdi_ellipses = [[1.0, diam, diam, 1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    phantom = odl.phantom.ellipse_phantom(projector.domain[0], ctdi_ellipses)
    phantom *= projector.domain[0].element(
        lambda x: (x[2] > z_offset) & (x[2] < z_offset + 150))
    phantom.show('phantom', coords=[None, None, z_offset + 75])

    den = phantom
    mat = phantom

    el = projector.domain.element([den, mat])
    result = projector(el)

    result.show('dose', cmap='jet', coords=[None, None, z_offset + 75])

    # Compute CTDI dose at center


    # Make indicator
    diam = 5.0 / 112.0  # relative size
    offset = 75.0 / 112.0  # relative size

    if dose_type == 'ctdi_center':
        # center
        ind_ellipses = [[1.0, diam, diam, 1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    elif dose_type == 'ctdi_peripheral':
        # edge, equal weight to all
        ind_ellipses = [[0.25, diam, diam, 1e6, offset, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.25, diam, diam, 1e6, -offset, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.25, diam, diam, 1e6, 0.0, offset, 0.0, 0.0, 0.0, 0.0],
                        [0.25, diam, diam, 1e6, 0.0, -offset, 0.0, 0.0, 0.0, 0.0]]
    elif dose_type == 'ctdi_w':
        # weighted mean, 1/3 of total should come from center, 1/6 for each edge
        ind_ellipses = [[1.0/3, diam, diam, 1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0/6, diam, diam, 1e6, offset, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0/6, diam, diam, 1e6, -offset, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0/6, diam, diam, 1e6, 0.0, offset, 0.0, 0.0, 0.0, 0.0],
                        [1.0/6, diam, diam, 1e6, 0.0, -offset, 0.0, 0.0, 0.0, 0.0]]
    elif dose_type == 'mean':
        # Mean over phantom
        ind_ellipses = ctdi_ellipses

    indicator = odl.phantom.ellipse_phantom(projector.domain[0], ind_ellipses)
    # Phantom is supposed to be 100 units long.
    indicator *= projector.domain[0].element(
        lambda x: (x[2] > z_offset + 25) & (x[2] < z_offset + 125))
    indicator.show('indicator', coords=[None, None, z_offset + 50])

    # Compute values
    J_per_kev = 1.6021766 * 10 ** -16  # Joule / KeV,  physical constant
    dose_J = np.sum(indicator * result) * J_per_kev  # J,  result is in KeV / voxel
    mass_kg = indicator.inner(den) / (10**3) ** 3  # density is kg / m^3, but indicator has units mm^3

    dose_per_mass = dose_J / mass_kg  # J / kg
    dose_mgy = dose_per_mass * 10 ** 3 # mJ / kg

    print('{} is {} mGy.'.format(dose_type, dose_mgy))

    # To get equivalent dose with more photons, simply scale linearly
    dose_scaled_mgy = dose_mgy * (332.0 / nProjection) * (15000 / photons_per_pixel)
    print('{} with 332 views at 50000 photons per pixel is {} mGy.'.format(dose_type, dose_scaled_mgy))
    # Typical values for imaging are 1-10 mGy for CBCT
