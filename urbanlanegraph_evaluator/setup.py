#/usr/bin/env python
from distutils.core import setup

setup(name='urbanlanegraph_evaluator',
      version='0.11',
      description='Evaluator for Lane Graph Predictions on the UrbanLaneGraph Dataset',
      author='Jannik Zuern, Martin Buechner',
      author_email='zuern@cs.uni-freiburg.de',
      url='http://urbanlanegraph.cs.uni-freiburg.de/',
      packages=['urbanlanegraph_evaluator',
                'urbanlanegraph_evaluator.metrics'],
      install_requires=[
          'numpy',
          'matplotlib',
          'networkx',
          'scipy',
          'shapely',
          'rtree',
          'opencv-python==4.7.0.72',
          'scikit-fmm',
          'utm',
          'pickle5',
      ]
      )

