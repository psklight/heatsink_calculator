import numpy as np

class Fluid:
    def __init__(self, density, viscosity, specific_heat, thermal_conductivity, viscosity_kinetic):
        self.density = density
        self.viscosity = viscosity
        self.specific_heat = specific_heat
        self.thermal_conductivity = thermal_conductivity
        self.viscosity_kinetic = viscosity_kinetic

    @property
    def prandtl(self):
        return self.viscosity * self.specific_heat / self.thermal_conductivity

class Solid:
    def __init__(self, thermal_conductivity):
        self.thermal_conductivity = thermal_conductivity

air = Fluid(density=1.177, viscosity=1.849e-5, specific_heat=1004.9, thermal_conductivity=0.02624,
           viscosity_kinetic=1.562e-5)