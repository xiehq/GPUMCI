# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

import odl
import GPUMCIPy
from gpumci import util

__all__ = ()


# TODO: make a class

def _get_phase_space(n):
    """Create phase space with n interactions."""
    pos = np.zeros([2 * n, 3])
    vel = np.zeros([2 * n, 3])
    vel[::2, :] = np.random.randn(n, 3)
    vel[::2, :] /= np.linalg.norm(vel[::2, :], axis=1)[:, None]
    vel[1::2, :] = -vel[::2, :]
    E = np.ones(2 * n) * 0.511
    return util.PhaseSpace(pos, vel, E)


if __name__ == '__main__':
    # Set geometry parameters
    volumeSize = np.array([224.0, 224.0, 224.0])
    volumeOrigin = np.array([-112.0, -112.0, -112.0])

    # Discretization parameters
    nVoxels = np.array([448, 448, 448])
    nProjection = 1

    # Continuous volume
    volume = odl.IntervalProd(volumeOrigin, volumeOrigin+volumeSize)

    phase_space = _get_phase_space(10**6)

    material_names = ['water']

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

    materials = util.getMaterialAttenuationTables(material_names)
    rayleighTables = util.getInteractionTables('rayleigh', material_names)
    comptonTables = util.getInteractionTables('compton', material_names)

    forward = GPUMCIPy.PhaseSpaceStorePhotonsMC(nVoxels, volumeOrigin, voxelSize,
                                                materials,
                                                rayleighTables, comptonTables)

    ind = 0
    phase_space_in = phase_space.getGPUCMIPhaseSpace()
    phase_space_out = phase_space.getGPUCMIPhaseSpace()

    den = odl.phantom.shepp_logan(reconDisc, modified=True)
    mat = reconMatDisc.zero()
    den[0] = 100

    forward.setData(den.ntuple.data_ptr, mat.ntuple.data_ptr)

    with odl.util.Timer():
        forward.project(phase_space_in, phase_space_out)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for i in range(0, 100, 2):
        if np.allclose(phase_space_out[i].direction,
                       -phase_space_out[i + 1].direction):
            c = 'r'
        else:
            c = 'b'

        plt.plot([phase_space_out[i].position[0], phase_space_out[i + 1].position[0]],
                 [phase_space_out[i].position[1], phase_space_out[i + 1].position[1]],
                 [phase_space_out[i].position[2], phase_space_out[i + 1].position[2]], c)

    ax.set_xlim([volume.min()[0], volume.max()[0]])
    ax.set_ylim([volume.min()[1], volume.max()[1]])
    ax.set_zlim([volume.min()[2], volume.max()[2]])
