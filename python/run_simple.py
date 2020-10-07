# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from math import sin, cos, pi

import numpy as np
import odl
import SimRec2DPy as SR
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

class CudaSimpleMCProjector(odl.Operator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volumeOrigin, voxelSize, nVoxels, nPixels, geometries, domain, range):
        self.geometries = geometries
        super().__init__(domain, range)
        
        self.forward = gpumci.SimpleMC(nVoxels, volumeOrigin, voxelSize, nPixels)

    def _apply(self, data, out):
        #Create projector
        mat = data.asarray()>0
        materials = odl.CudaEn(data.space.n, np.uint8).element(mat.flatten(order='F'))
        self.forward.setData(data.data_ptr, materials.data_ptr)

        out.set_zero()

        #Project all geometries
        for out_i, geo in zip(out, self.geometries):
            with Timer("projecting"):
                self.forward.project(geo.sourcePosition, geo.detectorOrigin, geo.pixelDirectionU, geo.pixelDirectionV, out_i[0].data_ptr, out_i[1].data_ptr)


#Set geometry parameters
volumeSize = np.array([200.0, 200.0, 200.0])
volumeOrigin = np.array([-100.0, -100.0, -100.0])

detectorSize = np.array([300.0, 300.0])
detectorOrigin = np.array([-150.0, -150.0])

sourceAxisDistance = 790.0
detectorAxisDistance = 210.0

#Discretization parameters
#nVoxels, nPixels = np.array([30 ,3,3]), np.array([3, 3])
nVoxels, nPixels = np.array([400, 400, 400]), np.array([100, 100])
nProjection = 4

#Scale factors
voxelSize = volumeSize / nVoxels
pixelSize = detectorSize / nPixels

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
dataDisc = odl.ProductSpace(odl.ProductSpace(projectionDisc,2), nProjection)

#Define the reconstruction space
reconSpace = odl.FunctionSpace(odl.Cuboid([0, 0, 0], volumeSize))

#Discretize the reconstruction space
reconRN = odl.CudaRn(nVoxels.prod())
reconDisc = odl.uniform_discr(reconSpace, reconRN, nVoxels, 'F')

#Create a phantom
phantom = SR.SRPyUtils.phantom(nVoxels[0:2])
phantom = np.repeat(phantom, nVoxels[-1]).reshape(nVoxels)
phantomVec = reconDisc.element(phantom)
plt.figure()
plt.imshow(phantom[:,:,nVoxels[2]//2])
plt.colorbar()

#Make the operator
projector = CudaSimpleMCProjector(volumeOrigin, voxelSize, nVoxels, nPixels, geometries, reconDisc, dataDisc)
result = projector(phantomVec)

nrows, ncols = 2, 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
fig.tight_layout()
figs, axess = plt.subplots(nrows=nrows, ncols=ncols)
figs.tight_layout()
for res, ax, axs in zip(result, axes.flat, axess.flat):
    print(res)
    ax.imshow(res[0].asarray().T, cmap='bone', origin='lower')
    ax.axis('off')
    axs.imshow(res[1].asarray().T, cmap='bone', origin='lower')
    axs.axis('off')

plt.show()
