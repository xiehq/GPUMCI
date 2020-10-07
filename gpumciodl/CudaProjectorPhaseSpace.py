# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

import odl
import GPUMCIPy
from gpumci import util

__all__ = ('CudaProjectorPhaseSpace',)


class CudaProjectorPhaseSpace(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volume, nVoxels, geometry, phase_space, n_runs,
                 material_names):

        self.geometry = geometry
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
        self.phase_space = phase_space

        materials = util.getMaterialAttenuationTables(material_names)
        rayleighTables = util.getInteractionTables('rayleigh', material_names)
        comptonTables = util.getInteractionTables('compton', material_names)

        self.forward = GPUMCIPy.PhaseSpaceMC(nVoxels, volumeOrigin, voxelSize,
                                             nPixels, n_runs, materials,
                                             rayleighTables, comptonTables)

        super().__init__(domain, range)

    def _call(self, data, out):
        # Create projector
        self.forward.setData(data[0].ntuple.data_ptr, data[1].ntuple.data_ptr)

        out.set_zero()

        # Project all geometries
        for ind, (out_i, geo) in enumerate(zip(out, self.geometries)):
            angle = float(self.geometry.motion_grid[ind])

            phase_space = self.phase_space.getGPUCMIPhaseSpace(
                self.geometry, angle)

            self.forward.project(geo.sourcePosition,
                                 geo.detectorOrigin,
                                 geo.pixelDirectionU,
                                 geo.pixelDirectionV,
                                 phase_space,
                                 out_i[0].ntuple.data_ptr,
                                 out_i[1].ntuple.data_ptr)


def _get_phase_space(n):
    pos = np.zeros([n, 3])
    pos[:, 0] = -1.0
    vel = np.zeros([n, 3])
    vel[:, 0] = 1.0
    E = np.ones(n) * 0.06
    return util.PhaseSpace(pos, vel, E)


if __name__ == '__main__':
    # Set geometry parameters
    volumeSize = np.array([224.0, 224.0, 224.0])
    volumeOrigin = np.array([-112.0, -112.0, -112.0])

    detectorSize = np.array([287.04, 264.94])
    detectorOrigin = np.array([-143.52, -132.0])

    sourceAxisDistance = 790.0
    detectorAxisDistance = 210.0

    # Discretization parameters
    nVoxels, nPixels = np.array([448, 448, 448]), np.array([78, 72])
    nProjection = 1

    # Continuous volume
    volume = odl.IntervalProd(volumeOrigin, volumeOrigin+volumeSize)

    # Geometry
    agrid = odl.uniform_partition(0, 0.0001 * np.pi, nProjection)
    dgrid = odl.uniform_partition(detectorOrigin, detectorOrigin+detectorSize,
                                  nPixels)
    geom = odl.tomo.CircularConeFlatGeometry(agrid, dgrid,
                                             src_radius=sourceAxisDistance,
                                             det_radius=detectorAxisDistance)

    phase_space = _get_phase_space(10000)

    materials = ['water']

    projector = CudaProjectorPhaseSpace(volume, nVoxels, geom, phase_space,
                                        10000, materials)

    den = odl.phantom.shepp_logan(projector.domain[0], False)
    mat = projector.domain[1].zero()
    result = projector([den, mat])

    index = 0
    result[index][0].show()
    result[index][1].show()
