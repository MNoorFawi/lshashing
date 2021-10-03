from setuptools import setup, Extension
import numpy as np

cython_exists = False

try:
    from Cython.Distutils import build_ext
    cython_exists = True
except ImportError:
    from distutils.command.build_ext import build_ext

if cython_exists:
    ext_modules=[
        Extension("pyxdist", ["lshashing/pyxdist.pyx", "lshashing/dist.c"], 
        include_dirs = [np.get_include()])
    ]

with open("README.md", "r", encoding="utf8") as rm:
    readme = rm.read()
    
with open("requirements.txt") as rq:
    requirements = rq.read().split('\n')

setup(
      name="lshashing",
      version="1.1.0",
      description="Nearest neighbors search using locality-sensitive hashing",
      packages=["lshashing"],
      install_requires=requirements,
      long_description=readme,
      long_description_content_type="text/markdown",
      url="https://github.com/MNoorFawi/lshashing",
      author="Muhammad N. Fawi",
      author_email="m.noor.fawi@gmail.com",
      license="MIT",
      classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
      cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules
      )
