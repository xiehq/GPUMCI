# -*- coding: utf-8 -*-
"""
File needed untill xraylib_np is available for numpy.

This wraps xraylib with vectorize instead.
"""

__all__ = ('xrl',)

import numpy as np
import xraylib

class xraylib_np_mock(object):
    def __getattr__(self, name):
        method = getattr(xraylib, name, None)
        if method is None:
            raise NotImplementedError('{} does not exist'.format(name))
        else:
            return np.vectorize(method)
    def __repr__(self):
        return 'xraylib_np_mock()'

xrl = xraylib_np_mock()