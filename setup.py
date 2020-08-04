from setuptools import setup, find_packages

setup(
    name='ipPlot',
    version='0.1.0',
    url='https://github.com/yooha1003/ipPlot',
    author='Uksu, Choi',
    author_email='qtwing@naver.com',
    description='Intensity profile of MR images plotting script',
    packages=find_packages(),
    install_requires=['matplotlib', 'numpy', 'pandas', 'nibabel', 'argparse', 'scipy','nilearn'],
)
