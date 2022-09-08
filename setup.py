from setuptools import setup

classifiers = []

with (open("README.md")) as f: 
  long_description = f.read()
 

setup(
  name="tracksim",
  url="",
  author="Luke Harwood",
  author_email="mwstudiodev@gmail.com",
  version="1.0.0",
  description="AI Track Simulator is an easy-to-use Simulator engine built in pygame",
  long_description=long_description,
  packages=["simulation"],
  include_package_data=True,
  license="MIT License",
  classifiers=classifiers,
  command_options={},
  python_requires=">=3.6",
  extras_require={
    "dev": ["pygame", "numpy"] 
  }
)
