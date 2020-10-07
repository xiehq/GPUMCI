# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

import odl

from gpumci import util, CudaProjectorGain

__all__ = ('CudaProjectorScatter', )


def sub_geometry(geometry, stride):
    angle_set = geometry.motion_partition.set
    angles = geometry.motion_partition.grid[::stride]
    angle_part = odl.RectPartition(angle_set, angles)

    return odl.tomo.HelicalConeFlatGeometry(angle_part,
                                            dpart=geometry.det_partition,
                                            src_radius=geometry.src_radius,
                                            det_radius=geometry.det_radius,
                                            pitch=geometry.pitch,
                                            axis=geometry.axis,
                                            src_to_det_init=geometry.src_to_det_init,
                                            det_init_axes=geometry.det_init_axes,
                                            pitch_offset=geometry.pitch_offset)

class CudaProjectorScatter(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volume, nVoxels, geometry,
                 gain, energies, spectrum, blur_radius, material_names):

        self.stride = 20
        geometries = util.getProjectionGeometries(geometry)
        inner_geometry = sub_geometry(geometry, self.stride)

        self._projector = CudaProjectorGain(volume, nVoxels, inner_geometry,
                                            gain, energies, energies, spectrum,
                                            material_names)

        self.n_inner = len(self._projector.range)

        # expected value of energy
        e_expected = np.sum(energies * spectrum)/np.sum(spectrum)
        self._scale = spectrum.sum() * e_expected
        self._proj_inner = self._projector.range.element()

        range = odl.ProductSpace(self._projector.range[0], len(geometries))
        super().__init__(self._projector.domain, range)

        self.gaussian = util.GaussianBlur(self._projector.range[0][0],
                                          blur_radius)

    def _call(self, data, out):
        # Create projector
        self._projector(data, self._proj_inner)

        self._proj_inner /= self._scale  # normalize by mean energy

        for proj in self._proj_inner:
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
    geom = odl.tomo.CircularConeFlatGeometry(agrid, dgrid,
                                             src_radius=790.0,
                                             det_radius=210.0)

    energies, spectrum = _get_spectrum(10)
    photons_per_pixel = 50
    spectrum *= photons_per_pixel
    gain = np.ones(nPixels)

    materials = ['air', 'water', 'bone']

    projector = CudaProjectorScatter(volume, nVoxels, geom, gain,
                                     energies, spectrum,
                                     blur_radius=[20, 30, 30],
                                     material_names=materials)

    den = odl.phantom.shepp_logan(projector.domain[0], True)
    mat = odl.phantom.shepp_logan(projector.domain[1])

    el = projector.domain.element([den, mat])
    el.show()
    result = projector(el)

    index = 0
    result[index][0].show()
    result[index][1].show()
