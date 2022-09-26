from setuptools import setup

classifiers = []

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pysimulate",
    url="https://github.com/lukeharwood11/ai-racecar-simulator",
    author="Luke Harwood",
    author_email="lukeharwood11@gmail.com",
    version="1.1.7",
    description="AI Track Simulator is an easy-to-use Simulator engine built in pygame",
    long_description=long_description,
    py_modules=["agent", "components", "genetic", "qlearn", "simulation", "utils", "vector2d", "vehicle"],
    long_description_content_type="text/markdown",
    keywords='simulator ai-gym racecar',
    package_dir={'': 'pysimulate'},
    include_package_data=True,
    license="MIT",
    classifiers=classifiers,
    command_options={},
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pygame",
        "sklearn",
        "scikit-learn",
        "tensorflow>=2.0.0",
    ]
)
