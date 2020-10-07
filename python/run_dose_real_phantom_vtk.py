# -*- coding: utf-8 -*-
from __future__ import division, print_function
from future import standard_library
standard_library.install_aliases()
from builtins import super

from math import sin, cos, pi, sqrt

import numpy as np
import odl

import GPUMCIPy as gpumci
from RL.util.testutils import Timer
import matplotlib.pyplot as plt

class ProjectionGeometry3D(object):
    """ Geometry for a specific projection
    """
    def __init__(self, sourcePosition, detectorOrigin, pixelDirectionU, pixelDirectionV):
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

    return gpumci.MaterialData(energyStep,compton,rayleigh,photo)

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
    densities = np.minimum(2.7,np.maximum(0.01,(phantomdata + 1000.0)/1000.0))
    densities = np.reshape(densities, [512, 512, 65], order='F')
    densities = densities[:,:,::-1]
    mat = np.minimum(2,np.round(densities))
    return densities, mat

class CudaSimpleMCProjector(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volumeOrigin, voxelSize, nVoxels, nPixels, geometries, domain, range, nphotons):
        self.geometries = geometries
        super().__init__(domain, range)
        
        materials = getMaterialInfo()
        rayleighTables = getRayleighTables()
        comptonTables = getComptonTables()
        with Timer('__init__'):
            self.forward = gpumci.DoseMC(nVoxels, volumeOrigin, voxelSize, nPixels, nphotons, materials, rayleighTables, comptonTables)

    def _apply(self, data, out):
        #Create projector
        with Timer('setData'):
            self.forward.setData(data[0].data_ptr, data[1].data_ptr)

        out.set_zero()

        #Project all geometries
        for out_i, geo in zip(out[0], self.geometries):
            for i in range(1):
                with Timer("projecting"):
                    self.forward.project(geo.sourcePosition, geo.detectorOrigin, geo.pixelDirectionU, geo.pixelDirectionV, out_i[0].data_ptr, out_i[1].data_ptr, out[1].data_ptr)

#Set geometry parameters
detectorSize = np.array([287.0, 265.0])
detectorOrigin = np.array([-143.5, -3.0])

sourceAxisDistance = 790.0
detectorAxisDistance = 210.0

#Discretization parameters
nVoxels, nPixels = np.array([512, 512, 65]), np.array([720,780])
nProjection = 4
nphotons = 100

#Scale factors
voxelSize = np.array([0.425,0.447,2.594])
pixelSize = detectorSize / nPixels

#Derived paramters
volumeSize = voxelSize*nVoxels
volumeOrigin = np.array([-volumeSize[0]/2.0, -volumeSize[1]/2.0, 0.0])

#Define projection geometries
geometries = []
angles = np.linspace(0, pi, nProjection, endpoint=False)
for theta in angles:
    x0 = np.array([cos(theta), sin(theta), 0.0])
    y0 = np.array([-sin(theta), cos(theta), 0.0])
    z0 = np.array([0.0, 0.0, 1.0])

    projSourcePosition = -sourceAxisDistance * x0
    projPixelDirectionU = y0 * pixelSize[0]
    projPixelDirectionV = z0 * pixelSize[1]
    projDetectorOrigin = detectorAxisDistance * x0 + detectorOrigin[0] * y0 + detectorOrigin[1] * z0
    geometries.append(ProjectionGeometry3D(projSourcePosition, projDetectorOrigin, projPixelDirectionU, projPixelDirectionV))

#Define the space of one projection
projectionSpace = odl.FunctionSpace(odl.Rectangle([0,0], detectorSize))
projectionRN = odl.CudaRn(nPixels.prod())

#Discretize projection space
projectionDisc = odl.uniform_discr(projectionSpace, projectionRN, nPixels, 'F')

#Define the reconstruction space
reconSpace = odl.FunctionSpace(odl.Cuboid([0, 0, 0], volumeSize))

#Discretize the reconstruction space
reconRN = odl.CudaRn(nVoxels.prod())
reconDisc = odl.uniform_discr(reconSpace, reconRN, nVoxels, 'F')

reconMatRN = odl.CudaEn(nVoxels.prod(), np.uint8)
reconMatDisc = odl.uniform_discr(reconSpace, reconMatRN, nVoxels, 'F')

reconProd = odl.productspace(reconDisc,reconMatDisc)

#Create the data space, which is the Cartesian product of the single projection spaces
dataDisc = odl.productspace(odl.ProductSpace(odl.ProductSpace(projectionDisc,2),
                                         nProjection),
                           reconDisc)

#Create a phantom
phantom,mat = getPhantom()

plt.figure()
plt.imshow(phantom[:,:,nVoxels[2]//2], cmap='bone')
plt.colorbar()

plt.figure()
plt.imshow(mat[:,:,nVoxels[2]//2], cmap='bone')
plt.colorbar()

materialsVec = reconMatDisc.element(mat)
phantomVec = reconDisc.element(phantom)
reconVec = reconProd.element(phantomVec,materialsVec)

#Make the operator
projector = CudaSimpleMCProjector(volumeOrigin, voxelSize, nVoxels, nPixels, geometries, reconProd, dataDisc, nphotons)
result = projector(reconVec)

if nProjection>1:
    nrows, ncols = int(sqrt(nProjection)), int(sqrt(nProjection))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.tight_layout()
    figs, axess = plt.subplots(nrows=nrows, ncols=ncols)
    figs.tight_layout()
    for res, ax, axs in zip(result[0], axes.flat, axess.flat):
        ax.imshow(res[0].asarray().T, cmap='bone', origin='lower')
        ax.axis('off')
        axs.imshow(res[1].asarray().T, cmap='bone', origin='lower')
        axs.axis('off')
else:
    res = result[0][0]
    plt.imshow(res[0].asarray().T, cmap='bone', origin='lower')
    plt.axis('off')
    plt.figure()
    plt.imshow(res[1].asarray().T, cmap='bone', origin='lower')
    plt.axis('off')

plt.figure()
dose_result = result[1].asarray()
plt.imshow(dose_result[:,:,nVoxels[2]//2], cmap='hot')
plt.colorbar()


from pyevtk.hl import gridToVTK

x = np.linspace(0, volumeSize[0], nVoxels[0]+1)
y = np.linspace(0, volumeSize[1], nVoxels[1]+1)
z = np.linspace(0, volumeSize[2], nVoxels[2]+1)

gridToVTK("./result", x, y, z, cellData = {"dose": np.ascontiguousarray(dose_result, np.float), "density" : np.ascontiguousarray(phantom, np.float),"material" : np.ascontiguousarray(mat, np.float), "doseperkg" : np.ascontiguousarray(dose_result/phantom)})

x = np.linspace(0, detectorSize[0], nPixels[0]+1)
y = np.linspace(0, detectorSize[1], nPixels[1]+1)
z = np.linspace(0, 180, nProjection+1)

primaries = np.dstack(tuple(res[0].asarray() for res in result[0]))
secondaries = np.dstack(tuple(res[1].asarray() for res in result[0]))

gridToVTK("./result_proj", x, y, z, cellData = {"primaries": np.ascontiguousarray(primaries, np.float), "secondaries" : np.ascontiguousarray(secondaries, np.float)})
plt.show()
