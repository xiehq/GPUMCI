# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import GPUMCIPy as gpumci


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

    return gpumci.MaterialData(energyStep, compton, rayleigh, photo);

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

#Set geometry parameters
detectorSize = np.array([287.0, 265.0])
detectorOrigin = np.array([-143.5, -3.0])

sourceAxisDistance = 790.0
detectorAxisDistance = 210.0

#Discretization parameters
nVoxels, nPixels = np.array([512, 512, 65]), np.array([720, 780])
nProjection = 1
nPhotons = 100

#Scale factors
voxelSize = np.array([0.425,0.447,2.594])
pixelSize = detectorSize / nPixels

#Derived paramters
volumeSize = voxelSize*nVoxels
volumeOrigin = np.array([-volumeSize[0]/2.0, -volumeSize[1]/2.0, 0.0])

#Define projection geometry
theta = 0.0
x0 = np.array([np.cos(theta), np.sin(theta), 0.0])
y0 = np.array([-np.sin(theta), np.cos(theta), 0.0])
z0 = np.array([0.0, 0.0, 1.0])

sourcePosition = -sourceAxisDistance * x0
pixelDirectionU = y0 * pixelSize[0]
pixelDirectionV = z0 * pixelSize[1]
detectorOrigin = detectorAxisDistance * x0 + detectorOrigin[0] * y0 + detectorOrigin[1] * z0

materials = getMaterialInfo()
rayleighTables = getRayleighTables()
comptonTables = getComptonTables()

forward = gpumci.GainMCSimple(nVoxels, volumeOrigin, 
                              voxelSize, nPixels, 
                              materials, rayleighTables, comptonTables)

#Create a phantom
phantom,mat = getPhantom()
plt.figure()
plt.imshow(phantom[:,:,nVoxels[2]//2], cmap='bone')
plt.colorbar()

plt.figure()
plt.imshow(mat[:,:,nVoxels[2]//2], cmap='bone')
plt.colorbar()

forward.setData(phantom.astype(dtype='float32', order='F'), 
                mat.astype(dtype='uint8', order='F'))

primary = np.zeros(nPixels, dtype='float32', order='F')
scatter = np.zeros(nPixels, dtype='float32', order='F')
gain = np.ones(nPixels, dtype='float32', order='F') * nPhotons

energyMin = 0.07
energyMax = 0.09

forward.project(sourcePosition, 
                detectorOrigin, 
                pixelDirectionU, 
                pixelDirectionV, 
                primary, 
                scatter,
                gain,
                energyMin,
                energyMax)

plt.figure()
plt.imshow(primary.T, origin='lower', interpolation='nearest')
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(scatter.T, origin='lower', interpolation='nearest')
plt.axis('off')
plt.show()
