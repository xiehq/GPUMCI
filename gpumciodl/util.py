# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import odl
import odlcuda.odlcuda_ as odlcuda
import os
import site
import GPUMCIPy

__all__ = ('ProjectionGeometry3D',
           'getMaterialAttenuationInfo', 'getMaterialAttenuationTables',
           'getInteractionTables', 'GaussianBlur', 'sub_geometry',
           'PhaseSpace')


def file_path(file):
    return os.path.join(site.getsitepackages()[0], 'gpumci', 'data', file)


class ProjectionGeometry3D(object):
    """ Geometry for a specific projection
    """
    def __init__(self, sourcePosition, detectorOrigin,
                 pixelDirectionU, pixelDirectionV):
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirectionU = pixelDirectionU
        self.pixelDirectionV = pixelDirectionV

    def __repr__(self):
        return ('ProjectionGeometry3D({}, {}, {}, {})'
                ''.format(self.sourcePosition,
                          self.detectorOrigin,
                          self.pixelDirectionU,
                          self.pixelDirectionV))


def getProjectionGeometries(geometry):
    angles = geometry.angles
    det_min = geometry.detector.params.min()
    pixel_size = geometry.detector.partition.cell_sides

    geometries = []
    for theta in angles:
        rotation_matrix = geometry.rotation_matrix(theta)

        projSourcePosition = geometry.src_position(theta)
        projPixelDirectionU = rotation_matrix.dot(geometry.detector.axes[0] * pixel_size[0])
        projPixelDirectionV = rotation_matrix.dot(geometry.detector.axes[1] * pixel_size[1])
        projDetectorOrigin = geometry.det_point_position(theta, det_min)
        geometries.append(ProjectionGeometry3D(
            projSourcePosition, projDetectorOrigin, projPixelDirectionU,
            projPixelDirectionV))

    return geometries


def getMaterialAttenuationInfo(energies, material_names):
    """ Get matrial attenuation info at energies.

    Materials default:

    ['air', 'water', 'bone', 'white_matter', 'grey_matter']
    """
    energies = np.asarray(energies)

    # Indata is by col: 0 energy, 1 rayleigh, 2 compton 3 photo
    materials = [np.loadtxt(file_path(mat + '.pdata'))
                 for mat in material_names]

    def interp(material_energies, material_value):
        return np.exp(np.atleast_2d(np.interp(energies,
                                              material_energies,
                                              np.log(material_value))))

    n_energies = energies.size

    compton = np.empty((0, n_energies))
    rayleigh = np.empty((0, n_energies))
    photo = np.empty((0, n_energies))

    for material in materials:
        material_energies = material[:, 0]
        compton = np.append(compton, interp(material_energies, material[:,2]), axis=0)
        rayleigh = np.append(rayleigh, interp(material_energies, material[:,1]), axis=0)
        photo = np.append(photo, interp(material_energies, material[:,3]), axis=0)

    return compton, rayleigh, photo


def getMaterialAttenuationTables(material_names):
    """ Get matrial attenuation info as GPUMCIPy table

    """
    n_energies = 1000
    energies = np.linspace(0, 0.2, n_energies)
    energyStep = energies[1] - energies[0]

    compton, rayleigh, photo = getMaterialAttenuationInfo(energies,
                                                          material_names)

    return GPUMCIPy.MaterialData(energyStep, compton, rayleigh, photo)



def getInteractionTables(interaction, material_names):
    """ Get matrial attenuation tables.

    Materials:

    ['air', 'water', 'bone', 'white_matter', 'grey_matter']
    """
    # info from files
    energyStep = 0.001

    table_data = GPUMCIPy.vector_array()

    for mat in material_names:
        filepath = '{}_{}_ne146_estep1kev.txt'.format(interaction, mat)
        table_data.append(np.loadtxt(file_path(filepath)))

    return GPUMCIPy.InteractionTables(energyStep, table_data)


class GaussianBlur(odl.Operator):
    def __init__(self, space, sigma):
        super().__init__(space, space)
        self.sigma = sigma
        self.tmp = space.element()

    def _call(self, data, out):
        odlcuda.gaussianBlur(data.ntuple.data,
                             self.tmp.ntuple.data,
                             out.ntuple.data,
                             data.shape[0],
                             data.shape[1],
                             self.sigma[0],
                             self.sigma[1],
                             int(self.sigma[0]*3),
                             int(self.sigma[1]*3))


class PhaseSpace(object):
    def __init__(self, positions, directions, energies):
        self.positions = positions
        self.directions = directions
        self.energies = energies
        self.weights = np.ones_like(energies)

    def getGPUCMIPhaseSpace(self, geometry=None, angle=None):
        if geometry is not None and angle is not None:
            rot = self.geometry.rotation_matrix(angle)
            src = self.geometry.src_position(angle)
            positions = src + rot.dot(self.positions.T).T
            directions = rot.dot(self.directions.T).T
        else:
            positions = self.positions.copy()
            directions = self.directions.copy()

        phase_space = GPUMCIPy.make_particle_array(positions,
                                                   directions,
                                                   self.energies,
                                                   self.weights)

        return phase_space


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
