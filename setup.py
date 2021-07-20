# Copyright 2015-2019 John Kitchin
# (see accompanying license files for details).
from setuptools import setup

setup(name='gespyranto',
      version='0.0.2',
      description='gespyranto',
      url='',
      maintainer='John Kitchin',
      maintainer_email='jkitchin@andrew.cmu.edu',
      license='GPL',
      platforms=['linux'],
      packages=['gespyranto'],
      setup_requires=[],
      data_files=[],
      install_requires=[],
      long_description='''
      ''')

# (shell-command "python setup.py register") to setup user
# to push to pypi - (shell-command "python setup.py sdist upload")


# Set TWINE_USERNAME and TWINE_PASSWORD in .bashrc
# python setup.py sdist bdist_wheel
# twine upload dist/*
