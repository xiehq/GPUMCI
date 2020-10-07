# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

import odl
import GPUMCIPy
from gpumci import util

__all__ = ('CudaProjectorSpectrum',)


class CudaProjectorSpectrum(odl.Operator):
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

        # Create the data space,
        # which is the Cartesian product of the single projection spaces
        range = odl.ProductSpace(odl.ProductSpace(projectionDisc, 2),
                                 len(self.geometries))

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

        # Set locals
        self.spectrum = [projectionDisc.element(g.copy()) for g in spectrum]
        self.energy_min = np.asarray(energy_min)
        self.energy_max = np.asarray(energy_max)

        materials = util.getMaterialAttenuationTables(material_names)
        rayleighTables = util.getInteractionTables('rayleigh', material_names)
        comptonTables = util.getInteractionTables('compton', material_names)

        self.forward = GPUMCIPy.GainMC(nVoxels, volumeOrigin, voxelSize,
                                       nPixels, materials, rayleighTables,
                                       comptonTables)

        super().__init__(domain, range)

    def _call(self, data, out):
        # Create projector
        self.forward.setData(data[0].ntuple.data_ptr, data[1].ntuple.data_ptr)

        out.set_zero()

        # Project all geometries
        for out_i, geo in zip(out, self.geometries):
            for i, spec in enumerate(self.spectrum):

                self.forward.project(geo.sourcePosition,
                                     geo.detectorOrigin,
                                     geo.pixelDirectionU,
                                     geo.pixelDirectionV,
                                     out_i[0].ntuple.data_ptr,
                                     out_i[1].ntuple.data_ptr,
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
    # Set geometry parameters
    volumeSize = np.array([224.0, 224.0, 224.0])
    volumeOrigin = np.array([-112.0, -112.0, 0.0])

    detectorSize = np.array([287.04, 264.94])
    detectorOrigin = np.array([-143.52, 0.0])

    sourceAxisDistance = 790.0
    detectorAxisDistance = 210.0

    # Discretization parameters
    nVoxels, nPixels = np.array([448, 448, 448]), np.array([780, 720])
    nProjection = 332

    # Continuous volume
    volume = odl.IntervalProd(volumeOrigin, volumeOrigin+volumeSize)

    # Geometry
    agrid = odl.uniform_partition(0, 1.2 * np.pi, nProjection)
    dgrid = odl.uniform_partition(detectorOrigin, detectorOrigin+detectorSize,
                                  nPixels)
    geom = odl.tomo.CircularConeFlatGeometry(agrid, dgrid,
                                             src_radius=sourceAxisDistance,
                                             det_radius=detectorAxisDistance)

    energies, spectrum = _get_spectrum(20)
    photons_per_pixel = 50
    spectrum *= photons_per_pixel
    gain = np.ones(nPixels)

    spectrum = np.tile(spectrum[:, None, None], [1, nPixels[0], nPixels[1]])

    materials = ['air', 'water', 'bone']

    projector = CudaProjectorSpectrum(volume, nVoxels, geom,
                                      energies, energies, spectrum, materials)

    den = odl.phantom.shepp_logan(projector.domain[0], True)
    mat = odl.phantom.shepp_logan(projector.domain[1])

    el = projector.domain.element([den, mat])
    el.show()
    result = projector(el)

    index = 0
    result[index][0].show()
    result[index][1].show()
