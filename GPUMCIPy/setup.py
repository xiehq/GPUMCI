# -*- coding: utf-8 -*-
"""

"""

from future import standard_library
standard_library.install_aliases()

from distutils.core import setup

setup(name='GPUMCIPy',
      version='0.1.0',
      author='Jonas Adler',
      author_email='jonasadl@kth.se',
      url='https://gits-14.sys.kth.se/jonasadl/GPUMCI',
      description='Python bindings for the GPUMCI library',
      license='Proprietary',
      packages=['GPUMCIPy'],
      package_dir={'GPUMCIPy': '.'},
      package_data={'GPUMCIPy': ['*.*']})
