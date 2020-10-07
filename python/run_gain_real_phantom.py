# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()

from math import sin, cos, pi, sqrt
import numpy as np

import odl
from odl.util.testutils import ProgressBar, Timer
import GPUMCIPy as gpumci

import matplotlib.pyplot as plt

class ProjectionGeometry3D(object):
    """ Geometry for a specific projection
    """
    def __init__(self, sourcePosition, detectorOrigin,
                 pixelDirectionU, pixelDirectionV):
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirectionU = pixelDirectionU
        self.pixelDirectionV = pixelDirectionV

def getMaterialInfo():
    airdata = np.loadtxt("../data/physics/attenuation/DryAir.pdata")
    waterdata = np.loadtxt("../data/physics/attenuation/Water.pdata")
    bonedata = np.loadtxt("../data/physics/attenuation/BoneCorticalIcru.pdata")

    # Indata is by col: 0 energy, 1 rayleigh, 2 compton 3 photo
    materials = (airdata, waterdata, bonedata)
    n_energies = 1000;
    energies = np.linspace(0,0.2,n_energies)
    energyStep = energies[1]-energies[0]

    compton = np.empty((0,n_energies))
    rayleigh = np.empty((0,n_energies))
    photo = np.empty((0,n_energies))

    for material in materials:
        compton = np.append(compton, np.atleast_2d(np.interp(energies,material[:,0],material[:,2])), axis=0)
        rayleigh = np.append(rayleigh, np.atleast_2d(np.interp(energies,material[:,0],material[:,1])), axis=0)
        photo = np.append(photo, np.atleast_2d(np.interp(energies,material[:,0],material[:,3])), axis=0)

    return gpumci.MaterialData(energyStep,compton,rayleigh*0,photo);

def getRayleighTables():
    #info from files
    energyStep = 0.001

    table_data=gpumci.vector_array()
    table_data.append(np.loadtxt('../data/physics/interaction/rayleigh_air_ne146_estep1kev.txt'))
    table_data.append(np.loadtxt('../data/physics/interaction/rayleigh_water_ne146_estep1kev.txt'))
    table_data.append(np.loadtxt('../data/physics/interaction/rayleigh_bone_ne146_estep1kev.txt'))
    return gpumci.InteractionTables(energyStep,table_data)

def getComptonTables():
    #info from files
    energyStep = 0.001

    table_data=gpumci.vector_array()
    table_data.append(np.loadtxt('../data/physics/interaction/compton_air_ne146_estep1kev.txt'))
    table_data.append(np.loadtxt('../data/physics/interaction/compton_water_ne146_estep1kev.txt'))
    table_data.append(np.loadtxt('../data/physics/interaction/compton_bone_ne146_estep1kev.txt'))
    return gpumci.InteractionTables(energyStep,table_data)

def getPhantom():
    phantomdata = np.loadtxt('../data/phantoms/davids_phantom.txt',delimiter=',');
    densities = np.minimum(2.7,np.maximum(0.0,(phantomdata + 1000.0)/1000.0))
    densities = np.reshape(densities, [512, 512, 65], order='F')
    densities = densities[:,:,::-1]
    mat = np.minimum(2,np.round(densities))
    return densities, mat

class CudaGainMCProjector(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """

    def __init__(self, volumeOrigin, voxelSize, nVoxels, nPixels, geometries, domain, range, gain):
        odl.Operator.__init__(self, domain, range)
        self.geometries = geometries

        self.gain = gain
        materials = getMaterialInfo()
        rayleighTables = getRayleighTables()
        comptonTables = getComptonTables()
        self.forward = gpumci.GainMC(nVoxels, volumeOrigin, voxelSize, nPixels, materials, rayleighTables, comptonTables)

    def _call(self, data, out):
        #Create projector
        with Timer('setData'):
            self.forward.setData(data[0].ntuple.data_ptr, data[1].ntuple.data_ptr)

        out.set_zero()

        nrun = 10

        energyMin = 0.05
        energyMax = 0.10

        progress = ProgressBar('Projecting', len(self.geometries), nrun)

        #Project all geometries
        for out_i, geo in zip(out, self.geometries):
            for i in range(nrun):
                self.forward.project(geo.sourcePosition,
                                     geo.detectorOrigin,
                                     geo.pixelDirectionU,
                                     geo.pixelDirectionV,
                                     out_i[0].ntuple.data_ptr,
                                     out_i[1].ntuple.data_ptr,
                                     self.gain.ntuple.data_ptr,
                                     energyMin,
                                     energyMax)

                progress.update()

#Set geometry parameters
detectorSize = np.array([287.0, 265.0])
detectorOrigin = np.array([-143.5, -3.0])

sourceAxisDistance = 790.0
detectorAxisDistance = 210.0

#Discretization parameters
nVoxels, nPixels = np.array([512, 512, 65]), np.array([720, 780])
nProjection = 1
nphotons = 100

#Scale factors
voxelSize = np.array([0.425,0.447,2.594])
pixelSize = detectorSize / nPixels

#Derived paramters
volumeSize = voxelSize*nVoxels
volumeOrigin = np.array([-volumeSize[0]/2.0, -volumeSize[1]/2.0, 0.0])

#Define projection geometries
geometries = []
for theta in np.linspace(0, pi, nProjection, endpoint=False):
    x0 = np.array([cos(theta), -sin(theta), 0.0])
    y0 = np.array([sin(theta), cos(theta), 0.0])
    z0 = np.array([0.0, 0.0, 1.0])

    projSourcePosition = -sourceAxisDistance * x0
    projPixelDirectionU = y0 * pixelSize[0]
    projPixelDirectionV = z0 * pixelSize[1]
    projDetectorOrigin = detectorAxisDistance * x0 + detectorOrigin[0] * y0 + detectorOrigin[1] * z0
    geometries.append(ProjectionGeometry3D(projSourcePosition, projDetectorOrigin, projPixelDirectionU, projPixelDirectionV))

#Define the space of one projection
projectionSpace = odl.FunctionSpace(odl.Rectangle([0,0], detectorSize))

#Discretize projection space
projectionDisc = odl.uniform_discr([0,0], detectorSize, nPixels,
                                   impl='cuda', dtype='float32', order='F')

#Create the data space, which is the Cartesian product of the single projection spaces
dataDisc = odl.ProductSpace(odl.ProductSpace(projectionDisc, 2), nProjection)

#Define the reconstruction space
reconDisc = odl.uniform_discr([0, 0, 0], volumeSize, nVoxels,
                              impl='cuda', dtype='float32', order='F')

reconMatDisc = odl.uniform_discr([0, 0, 0], volumeSize, nVoxels,
                                 impl='cuda', dtype='uint8', order='F')

reconProd = odl.ProductSpace(reconDisc, reconMatDisc)

#Create a phantom
phantom,mat = getPhantom()
plt.figure()
plt.imshow(phantom[:, :, nVoxels[2]//2], cmap='bone')
plt.colorbar()

plt.figure()
plt.imshow(mat[:, :, nVoxels[2]//2], cmap='bone')
plt.colorbar()

reconVec = reconProd.element([phantom, mat])

gain_data = np.exp((projectionDisc.points()[:,0] - detectorSize[0]/2)**2 / 135**2)*nphotons
gain_image = projectionDisc.element(gain_data)

#Make the operator
projector = CudaGainMCProjector(volumeOrigin, voxelSize, nVoxels, nPixels, geometries, reconProd, dataDisc, gain_image)
result = projector(reconVec)

def as_image(vector):
    return np.rot90(vector.asarray())

if nProjection>1:
    nrows, ncols = int(sqrt(nProjection)), int(sqrt(nProjection))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.tight_layout()
    figs, axess = plt.subplots(nrows=nrows, ncols=ncols)
    figs.tight_layout()
    for res, ax, axs in zip(result, axes.flat, axess.flat):
        ax.imshow(-np.log(as_image(res[0])), cmap='bone')
        ax.axis('off')
        axs.imshow(as_image(res[1]), cmap='bone')
        axs.axis('off')
else:
    res = result[0]
    plt.figure()
    plt.imshow(as_image(res[0]), cmap='CMRmap')
    plt.axis('off')
    plt.figure()
    plt.imshow(as_image(res[1]), cmap='CMRmap')
    plt.axis('off')
plt.show()
