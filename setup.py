from setuptools import setup, find_packages

with open('requirements.txt') as fr:
    required = fr.read().splitlines()

setup(
        name='yolo',
        version='0.1.0',
        install_requires=required,
        packages=find_packages(),
        url='',
        author='jianantian',
        author_email='emile.zhu@hotmail.com',
        zip_safe=False,
        description='object detection by yolo3'
)