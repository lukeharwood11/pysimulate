from setuptools import setup, find_packages

classifiers = []

with (open("README.md")) as f: 
  long_description = f.read()
 

setup(
  name="ai_racecar_simulator",
  url="https://github.com/lukeharwood11/ai-racecar-simulator",
  author="Luke Harwood",
  author_email="mwstudiodev@gmail.com",
  version="1.0.2",
  description="AI Track Simulator is an easy-to-use Simulator engine built in pygame",
  long_description=long_description,
  packages=find_packages("org.mwdev.simulator"),
  keywords='simulator ai-gym racecar',
  package_dir={'': 'org.mwdev.simulator'},
  include_package_data=True,
  license="MIT",
  classifiers=classifiers,
  command_options={},
  python_requires=">=3.6",
  install_requires=[
    'numpy', 'tensorflow', 'pygame', 'keras'
  ]
)
