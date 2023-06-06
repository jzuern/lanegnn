from setuptools import setup

setup(
   name='LaneGNN',
   version='0.2',
   author='Martin Buechner, Jannik Zuern, both University of Freiburg',
   author_email='buechner@cs.uni-freiburg.de',
   packages=['lanegnn',
             'lanegnn.inference',
             'lanegnn.learning',
             'lanegnn.utils',
             ],
   description='Implementation of the LaneGNN method for successor lane graph prediction as well as an aggregation method for lane graph aggregation.',
)
