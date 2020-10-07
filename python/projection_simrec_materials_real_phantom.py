# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from math import sin, cos, pi, sqrt

import numpy as np
import odl

import SimRec2DPy as SR
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

def getPhantom():
    phantomdata = np.loadtxt('../data/phantoms/davids_phantom.txt',delimiter=',');
    densities = np.minimum(2.7,np.maximum(0.0,(phantomdata + 1000.0)/1000.0))
    densities = np.reshape(densities, [512, 512, 65], order='F')
    densities = densities[:,:,::-1]
    mat = np.minimum(2,np.round(densities))
    return densities, mat

class CudaProjector3D(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volumeOrigin, voxelSize, nVoxels, nPixels, geometries, domain, range, stepSize):
        self.geometries = geometries
        self.forward = SR.SRPyCuda.CudaForwardProjector3D(nVoxels, volumeOrigin, voxelSize, nPixels, stepSize)
        super().__init__(domain, range, True)

    def _apply(self, data, out):
        #Create projector
        self.forward.setData(data.data_ptr)

        #Project all geometries
        for i in range(len(self.geometries)):
            geo = self.geometries[i]
            self.forward.project(geo.sourcePosition, geo.detectorOrigin, geo.pixelDirectionU, geo.pixelDirectionV, out[i].data_ptr)

#Set geometry parameters

detectorSize = np.array([287.0, 265.0])
detectorOrigin = np.array([-143.5, -3.0])

sourceAxisDistance = 790.0
detectorAxisDistance = 210.0

#Discretization parameters
nVoxels, nPixels = np.array([512, 512, 65]), np.array([720,780])
nProjection = 4
nphotons = 300

#Scale factors
voxelSize = np.array([0.425,0.447,2.594])
pixelSize = detectorSize / nPixels

#Derived paramters
volumeSize = voxelSize*nVoxels
volumeOrigin = np.array([-volumeSize[0]/2.0, -volumeSize[1]/2.0, 0.0])

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
projectionRN = odl.CudaRn(nPixels.prod())

#Discretize projection space
projectionDisc = odl.uniform_discr(projectionSpace, projectionRN, nPixels, 'F')

#Create the data space, which is the Cartesian product of the single projection spaces
dataDisc = odl.ProductSpace(projectionDisc, nProjection)

#Define the reconstruction space
reconSpace = odl.FunctionSpace(odl.Cuboid([0, 0, 0], volumeSize))

#Discretize the reconstruction space
reconRN = odl.CudaRn(nVoxels.prod())
reconDisc = odl.uniform_discr(reconSpace, reconRN, nVoxels, 'F')

#Create a phantom
phantom,mat = getPhantom()
plt.figure()
plt.imshow(phantom[:,:,nVoxels[2]//2], cmap='bone')
plt.colorbar()

plt.figure()
plt.imshow(mat[:,:,nVoxels[2]//2], cmap='bone')
plt.colorbar()

phantomVec = reconDisc.element(phantom)

#Make the operator
stepSize = voxelSize.min()
projector = CudaProjector3D(volumeOrigin, voxelSize, nVoxels, nPixels, geometries, reconDisc, dataDisc, stepSize)
with Timer('project'):
    result = projector(phantomVec)

if nProjection>1:
    nrows, ncols = int(sqrt(nProjection)), int(sqrt(nProjection))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.tight_layout()
    for res, ax in zip(result, axes.flat):
        ax.imshow(res.asarray().T, cmap='bone', origin='lower')
        ax.axis('off')
else:
    res = result[0]
    plt.imshow(res.asarray().T, cmap='bone', origin='lower')
    plt.axis('off')
plt.show()
