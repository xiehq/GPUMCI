# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

import odl
try:
    import SimRec2DPy as SR
except ImportError:
    SR = 'SimRec2DPy not installed'
    pass
from gpumci import util

__all__ = ('CudaProjectorPrimary',)


def _getMaterialInfo(material_names):
    n_energies = 1000
    energies = np.linspace(0, 0.2, n_energies)

    compton, rayleigh, photo = util.getMaterialAttenuationInfo(energies,
                                                               material_names)

    att = compton + rayleigh + photo
    return energies, att


class CosWeighting(object):
    def __init__(self, nPixels, geometries):
        self.geometries = geometries
        self._cosweighting = SR.SRPyCuda.CudaCosWeighting(nPixels)

    def __call__(self, inputs):
        for inp, geo in zip(inputs, self.geometries):
            self._cosweighting.apply(geo.sourcePosition,
                                     geo.detectorOrigin,
                                     geo.pixelDirectionU,
                                     geo.pixelDirectionV,
                                     inp.ntuple.data_ptr,
                                     inp.ntuple.data_ptr)


class CudaProjectorPrimary(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volume, nVoxels, geometry, stepSize,
                 gain, energies, spectrum, material_names):
        self.stepSize = stepSize

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

        self.forward = SR.SRPyCuda.CudaForwardProjectorIndexed3D(nVoxels,
                                                                 volumeOrigin,
                                                                 voxelSize,
                                                                 nPixels,
                                                                 stepSize)

        ran = odl.ProductSpace(projectionDisc, len(self.geometries))
        super().__init__(domain, ran)

        # Specifics for this application
        self.n_materials = len(material_names)
        self._spectrum_energies = np.asarray(energies)
        self._spectrum = np.asarray(spectrum) / sum(spectrum)
        self._intensities = self._spectrum * self._spectrum_energies
        self._intensities *= sum(self._spectrum) / sum(self._intensities)

        self._energies, self._att = _getMaterialInfo(material_names)
        self._projec_accum = self.range[0].element()
        self._projec_mat = [self.range[0].element()
                            for i in range(self.n_materials)]

        # cos weighting
        self._cosweight = CosWeighting(nPixels, self.geometries)

        # gain corr
        self._gain = self.range[0].element(gain)

    def _call(self, data, out):
        self.forward.setData(data[0].ntuple.data_ptr, data[1].ntuple.data_ptr)

        for i in range(len(out)):
            geo = self.geometries[i]
            for index in range(self.n_materials):  # for each material, project
                self.forward.project(geo.sourcePosition, geo.detectorOrigin,
                                     geo.pixelDirectionU, geo.pixelDirectionV,
                                     index, self._projec_mat[index].ntuple.data_ptr)
                # scale by step to get lintegral of density along line
                self._projec_mat[index] *= self.stepSize

            # Accumulate result in projections[i]
            out[i].set_zero()
            for energy, intensity in zip(self._spectrum_energies,
                                         self._intensities):
                self._projec_accum.set_zero()
                # for each material, accumulate number of mfps
                for index in range(self.n_materials):
                    coeff = np.interp(energy,
                                      self._energies,
                                      self._att[index, :]) / 10.0  # mm -> cm

                    # self._projec_accum += coeff * self._projec_mat[index-1]
                    self._projec_accum.lincomb(1, self._projec_accum,
                                               coeff, self._projec_mat[index])

                # exp(-x)
                self._projec_accum *= -1
                self._projec_accum.ufunc.exp(out=self._projec_accum)

                # accumulate with linear response in energy
                out[i].lincomb(1, out[i],
                               intensity, self._projec_accum)

            out[i] *= self._gain

        # cos weight
        self._cosweight(out)

def _get_spectrum(n):
    """ A simple example spectrum """
    energies = np.linspace(0.03, 0.09, n)
    #polynomial with zero at E=0 and E=0.09, peak at E=0.06
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

    stepsize = (volume.extent() / nVoxels).min() / 1.0

    energies, spectrum = _get_spectrum(5)
    gain = np.ones(nPixels)

    materials = ['air', 'water', 'bone']

    projector = CudaProjectorPrimary(volume, nVoxels, geometry, stepsize, gain,
                                     energies, spectrum, materials)

    den = odl.phantom.shepp_logan(projector.domain[0], True)
    mat = odl.phantom.shepp_logan(projector.domain[1])
    el = projector.domain.element([den, mat])
    result = projector(el)

    result[0].show()
