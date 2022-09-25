# pysimulate

```
pip install pysimulate
```
## Purpose

This project provides a framework to build a simulator to test different agents in a simple environment. This allows you to control the inputs/outputs of the system in python and hotswap the 'drivers' to the vehicles.

Use this repo to build your own AI drivers and test them on your own tracks!

## Links

- [pygame docs](https://www.pygame.org/docs/)

## About

In order to install dependencies for local development (using `setup.py`), use the command:
```
pip install -e .
```

Extend from the abstract class `Simulation` from `simulation.py` in order to create a custom simulation or simply use the `DefaultSimulation` class.

Same applies to building a custom vehicle from `vehicle.py` or using the `DefaultVehicle` class.

## Author

**Luke Harwood** 

_lukeharwood11@gmail.com_

_harwoodl@msoe.edu_ 

_www.lukeharwood.dev_

Created on 05/24/2022

