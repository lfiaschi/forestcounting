from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
pyxfolder="cythonf/"

ext_modules = [Extension(
                         "_sampling",
                        
                          [pyxfolder+"_sampling.pyx"],
                          include_dirs=[np.get_include()]
                          ),

                Extension(
                         "_spatialAverage",
                        
                          [pyxfolder+"_spatialAverage.pyx"],
                          include_dirs=[np.get_include()]
                          ),
                          
                Extension(
                         "_helpers",
                        
                          [pyxfolder+"_helpers.pyx"],
                          include_dirs=[np.get_include()]
                          )
               ]

setup(
  name = '',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)