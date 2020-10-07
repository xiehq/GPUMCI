# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from math import sin, cos, pi, sqrt

import numpy as np
import odl
import GPUMCIPy as gpumci
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

    return gpumci.MaterialData(energyStep, compton, rayleigh, photo)

def getPhantom():
    phantomdata = np.loadtxt('../data/phantoms/davids_phantom.txt',delimiter=',');
    densities = np.minimum(2.7,np.maximum(0.0,(phantomdata + 1000.0)/1000.0))
    densities = np.reshape(densities, [512, 512, 65], order='F')
    densities = densities[:,:,::-1]
    mat = np.minimum(2,np.round(densities))
    return densities, mat
    
class CudaSimpleMCProjector(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volumeOrigin, voxelSize, nVoxels, nPixels, geometries, domain, range, energy, steplen):
        self.geometries = geometries
        super().__init__(domain, range)
        self.energy = energy
        self.steplen = steplen
        materials = getMaterialInfo()
        with odl.util.Timer('__init__'):
            self.forward = gpumci.ForwardProjector(nVoxels, volumeOrigin, 
                                                   voxelSize, nPixels, 
                                                   materials)

    def _apply(self, data, out):
        #Create projector
        with odl.util.Timer('setData'):
            self.forward.setData(data[0].ntuple.data_ptr, data[1].ntuple.data_ptr)

        out.set_zero()

        #Project all geometries
        for out_i, geo in zip(out, self.geometries):
            with odl.util.Timer("projecting"):
                self.forward.project(geo.sourcePosition, geo.detectorOrigin, 
                                     geo.pixelDirectionU, geo.pixelDirectionV,
                                     self.energy, self.steplen,
                                     out_i.ntuple.data_ptr)

#Set geometry parameters
detectorSize = np.array([287.0, 265.0])
detectorOrigin = np.array([-143.5, -3.0])

sourceAxisDistance = 790.0
detectorAxisDistance = 210.0

#Discretization parameters
nVoxels, nPixels = np.array([512, 512, 65]), np.array([720, 780])
nProjection = 1

#Scale factors
voxelSize = np.array([0.425, 0.447, 2.594])
pixelSize = detectorSize / nPixels

#Derived paramters
volumeSize = voxelSize*nVoxels
volumeOrigin = np.array([-volumeSize[0]/2.0, -volumeSize[1]/2.0, 0.0])

#Simulation factors
steplen = voxelSize.min()
energy = 0.09 #MeV

#Define projection geometries
geometries = []
for theta in np.linspace(0, pi, nProjection, endpoint=False):
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

#Discretize projection space
projectionDisc = odl.uniform_discr(projectionSpace, nPixels, 
                                           impl='cuda',
                                           order='F')

#Create the data space, which is the Cartesian product of the single projection spaces
dataDisc = odl.ProductSpace(projectionDisc, nProjection)

#Define the reconstruction space
reconSpace = odl.FunctionSpace(odl.Cuboid([0, 0, 0], volumeSize))

#Discretize the reconstruction space
reconDisc = odl.uniform_discr(reconSpace, nVoxels, 
                                      impl='cuda',
                                      order='F')
reconMatDisc = odl.uniform_discr(reconSpace, nVoxels, 
                                         impl='cuda', 
                                         dtype='uint8',
                                         order='F')

reconProd = odl.ProductSpace(reconDisc,reconMatDisc)

#Create a phantom
phantom,mat = getPhantom()

plt.figure()
plt.imshow(phantom[:,:,nVoxels[2]//2], cmap='bone')
plt.colorbar()

plt.figure()
plt.imshow(mat[:,:,nVoxels[2]//2], cmap='bone')
plt.colorbar()

reconVec = reconProd.element([phantom, mat])

#Make the operator
projector = CudaSimpleMCProjector(volumeOrigin, voxelSize, 
                                  nVoxels, nPixels, geometries, 
                                  reconProd, dataDisc, energy, steplen)
result = projector(reconVec)

def as_image(vector):
    return vector.asarray().T
    
def as_log_image(vector):
    arr = vector.asarray().T
    return -np.log(arr/arr.max())

if nProjection>1:
    nrows, ncols = int(sqrt(nProjection)), int(sqrt(nProjection))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.tight_layout()
    for res, ax in zip(result, axes.flat):
        ax.imshow(as_log_image(res), origin='lower')
        ax.axis('off')
else:
    res = result[0]
    plt.figure()
    plt.imshow(as_log_image(res), origin='lower')
    plt.colorbar()
    plt.axis('off')
plt.show()
