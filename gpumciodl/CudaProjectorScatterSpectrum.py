# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

import odl

from gpumci import util, CudaProjectorSpectrum

__all__ = ('CudaProjectorScatterSpectrum', )


class CudaProjectorScatterSpectrum(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volume, nVoxels, geometry,
                 energies, spectrum, blur_radius, material_names):

        self.stride = blur_radius[0]
        geometries = util.getProjectionGeometries(geometry)
        inner_geometry = util.sub_geometry(geometry, self.stride)
        self._projector = CudaProjectorSpectrum(volume, nVoxels, inner_geometry,
                                                energies, energies, spectrum,
                                                material_names)

        # scale the values so that the value of unattenuated results is 1.0

        # expected value of energy
        e_expected = np.sum(energies[:, None, None] * spectrum, axis=0) / np.sum(spectrum, axis=0)
        self._scale = self._projector.range[0][0].element(e_expected)
        self._proj_inner = self._projector.range.element()

        range = odl.ProductSpace(self._projector.range[0], len(geometries))
        super().__init__(self._projector.domain, range)

        if blur_radius is not None:
            self.gaussian = util.GaussianBlur(self._projector.range[0][0],
                                              blur_radius[1:])
        else:
            self.gaussian = None

    def _call(self, data, out):
        # Create projector
        self._projector(data, self._proj_inner)

        for proj in self._proj_inner:
            proj[0] /= self._scale  # normalize by mean energy
            proj[1] /= self._scale  # normalize by mean energy

            if self.gaussian is not None:
                self.gaussian(proj[1], proj[1])

        for i in range(len(out)):
            previous = self._proj_inner[i//self.stride]
            next = self._proj_inner[min(len(self._proj_inner)-1,
                                        i//self.stride+1)]

            # Linear interpolation
            x = i/self.stride - i//self.stride
            out[i].lincomb(1.0-x, previous, x, next)


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

    materials = ['air', 'water', 'bone']

    spectrum = np.tile(spectrum[:, None, None], [1, nPixels[0], nPixels[1]])

    projector = CudaProjectorScatterSpectrum(volume, nVoxels,
                                             geometry, energies, spectrum,
                                             blur_radius=[20, 30, 30],
                                             material_names=materials)

    den = odl.phantom.shepp_logan(projector.domain[0], True)
    mat = odl.phantom.shepp_logan(projector.domain[1])

    el = projector.domain.element([den, mat])
    el.show()
    result = projector(el)

    index = 10
    result[index][0].show()
    result[index][1].show()
