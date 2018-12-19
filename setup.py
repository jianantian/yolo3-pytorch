from setuptools import setup

with open('requirements.txt') as fr:
    required = fr.read().splitlines()

setup(
    name='yolo',
    version='0.1.0',
    install_requires=required,
    packages=['yolo', 'resource', 'cfg'],
    include_package_data=True,
    url='',
    author='emile',
    author_email='emile.zhu@hotmail.com',
    zip_safe=False,
    license='BSD 3-Clause',
    description='object detection by yolo3'
)
