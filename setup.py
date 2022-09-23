from setuptools import setup, find_packages

classifiers = []

with (open("description.txt")) as f:
  long_description = f.read()
 

setup(
  name="pysimulate",
  url="https://github.com/lukeharwood11/ai-racecar-simulator",
  author="Luke Harwood",
  author_email="mwstudiodev@gmail.com",
  version="1.1.0",
  description="AI Track Simulator is an easy-to-use Simulator engine built in pygame",
  long_description=long_description,
  packages=["pysimulate"],
  keywords='simulator ai-gym racecar',
  package_dir={'': 'org.mwdev'},
  include_package_data=True,
  license="MIT",
  classifiers=classifiers,
  command_options={},
  python_requires=">=3.6",
  install_requires=[
    'numpy', 'tensorflow', 'pygame', 'keras', 'sklearn', 'scikit-learn'
  ]
)
