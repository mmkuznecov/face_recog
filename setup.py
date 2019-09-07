from setuptools import setup, find_packages
from os.path import join, dirname

setup(
    name = 'face_recog',
    version = 1.0,
    packages = ['face_recog'],
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    entry_points={'console_scripts':['train_encoder = face_recog.face_recog:train_encoder']}
)