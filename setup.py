from setuptools import setup

setup(name='thermometr',
      version='0.1',
      description='The Anomaly Detection Package',
      url='https://github.com/nlittlepoole/thermometr',
      author='Niger Little-Poole',
      author_email='nlittlepoole@gmail.com',
      license='MIT',
      packages=['thermometr'],
      install_requires=[
        "statsmodels",
        "numpy",
        "pandas",
        "scipy"
      ],
      zip_safe=False)
