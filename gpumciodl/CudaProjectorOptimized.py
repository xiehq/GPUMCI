# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

import odl

from gpumci import (ProjectionGeometry3D, CudaProjectorPrimary,
                    CudaProjectorScatter)

__all__ = ('ProjectionGeometry3D', 'CudaProjectorOptimized')


class CudaProjectorOptimized(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volume, nVoxels, geometry, stepSize,
                 gain, energies, spectrum,
                 blur_radius, material_names):
        self.projector_primary = CudaProjectorPrimary(volume, nVoxels,
                                                      geometry, stepSize,
                                                      gain, energies, spectrum,
                                                      material_names=material_names)
        self.projector_scatter = CudaProjectorScatter(volume, nVoxels,
                                                      geometry, gain,
                                                      energies, spectrum,
                                                      blur_radius,
                                                      material_names=material_names)

        # we dont need a result for the primaries
        self.tmp_scatter = self.projector_primary.range.element()
        domain = self.projector_primary.domain
        range = self.projector_primary.range
        super().__init__(domain, range)

    def _call(self, volume, out):
        # we use the projections data as a temporary
        tmp = self.projector_scatter.range.element(zip(out,
                                                       self.tmp_scatter))

        self.projector_scatter(volume, tmp)
        self.projector_primary(volume, out)

        # add scatter
        for prim, sca in zip(out, self.tmp_scatter):
            prim += sca


def _get_spectrum(n):
    energies = np.linspace(0.03, 0.09, n)
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

    # Discretization parameters
    nVoxels, nPixels = np.array([448, 448, 448]), np.array([780, 720])
    nProjection = 332

    # Continuous volume
    volume = odl.IntervalProd(volumeOrigin, volumeOrigin+volumeSize)

    # Geometry
    agrid = odl.uniform_partition(0, 1.2 * np.pi, nProjection)
    dgrid = odl.uniform_partition(detectorOrigin, detectorOrigin+detectorSize,
                                  nPixels)
    geometry = odl.tomo.CircularConeFlatGeometry(agrid, dgrid,
                                                 src_radius=790.0,
                                                 det_radius=210.0)

    energies, spectrum = _get_spectrum(10)
    photons_per_pixel = 50
    spectrum *= photons_per_pixel
    gain = np.ones(nPixels)

    materials = ['air', 'water', 'bone']
    stepsize = (volume.extent() / nVoxels).min() / 1.0

    projector = CudaProjectorOptimized(volume, nVoxels, geometry,
                                       stepsize, gain, energies, spectrum,
                                       blur_radius=[20, 30, 30],
                                       material_names=materials)

    den = odl.phantom.shepp_logan(projector.domain[0], True)
    mat = odl.phantom.shepp_logan(projector.domain[1])
    el = projector.domain.element([den, mat])
    result = projector(el)

    index = 10
    result[index].show()
