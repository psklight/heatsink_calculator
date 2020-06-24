import numpy as np
from .materials import *


def nusselt(reynolds, prandtl):
    a = 8/np.power(reynolds*prandtl, 3)
    b = 1/np.power(0.664*np.sqrt(reynolds)*np.power(prandtl, 1.0/3)*np.sqrt(1+3.65/np.sqrt(reynolds)),
                 3)
    return np.power(a+b, -1.0/3)


class HeatsinkBasic:

    def __init__(self, base_width, base_height, base_k, fin_height, fin_width, fin_gap, fin_k, length):
        self.base_width = base_width
        self.base_height = base_height
        self.base_k = base_k
        self.fin_height = fin_height
        self.fin_width = fin_width
        self.fin_k = fin_k
        self.fin_gap = fin_gap
        self.length = length

    @property
    def fin_corrected_height(self):
        """
        From Cengel's Intro to Thermodynamics and Heat Transfer, Chapter 10. For rectangular fins.
        """
        return self.fin_height + self.fin_width/2

    @property
    def fin_number(self):
        n = np.floor(self.base_width / (self.fin_gap + self.fin_width))
        left_over = self.base_width - n * (self.fin_gap + self.fin_width)
        if left_over >= self.fin_width:
            n += 1
        return int(n)

    @property
    def base_exposed_surface(self):
        return (self.base_width - self.fin_width*self.fin_number) * self.length

    @property
    def fin_exposed_surface(self):
        return 2.0 * self.fin_height * self.length

    @property
    def flow_area(self):
        return (self.base_width - self.fin_number*self.fin_width) * self.fin_height

    @property
    def base_thermal_resistance(self):
        return self.base_height / self.base_k / self.base_width / self.length

    @property
    def free_area_ratio(self):
        return 1 - self.fin_width * self.fin_number / self.base_width

    @property
    def flow_contraction_coeff(self):
        free_area_ratio = self.free_area_ratio
        return 0.42 * (1 - free_area_ratio ** 2)

    @property
    def flow_expansion_coeff(self):
        free_area_ratio = self.free_area_ratio
        return (1 - free_area_ratio ** 2) ** 2

    @property
    def hydraulic_diameter(self):
        """
        https://www.electronics-cooling.com/2003/05/estimating-parallel-plate-fin-heat-sink-pressure-drop/
        Dh ~ 2*fin_gap
        """
        return 2 * self.fin_gap

    @property
    def flow_aspect_ratio(self):
        return self.fin_gap / self.fin_height


class HeatsinkBasicwFinNumber:

    def __init__(self, base_width, base_height, base_k, fin_height, fin_width, fin_k, fin_number, length):
        self.base_width = base_width
        self.base_height = base_height
        self.base_k = base_k
        self.fin_height = fin_height
        self.fin_width = fin_width
        self.fin_k = fin_k
        self.fin_number = fin_number
        self.length = length

    @property
    def fin_gap(self):
        return (self.base_width - self.fin_number * self.fin_width) / (self.fin_number - 1.0)

    @property
    def base_exposed_surface(self):
        return (self.base_width - self.fin_number * self.fin_width) * self.length

    @property
    def fin_exposed_surface(self):
        return 2.0 * self.fin_height * self.length

    @property
    def flow_area(self):
        return (self.base_width - self.fin_width * self.fin_number) * self.fin_height

    @property
    def base_thermal_resistance(self):
        return self.base_height / self.base_k / self.base_width / self.length

    @property
    def free_area_ratio(self):
        return 1 - self.fin_width * self.fin_number / self.base_width

    @property
    def flow_contraction_coeff(self):
        free_area_ratio = self.free_area_ratio
        return 0.42 * (1 - free_area_ratio ** 2)

    @property
    def flow_expansion_coeff(self):
        free_area_ratio = self.free_area_ratio
        return (1 - free_area_ratio ** 2) ** 2

    @property
    def hydraulic_diameter(self):
        """
        https://www.electronics-cooling.com/2003/05/estimating-parallel-plate-fin-heat-sink-pressure-drop/
        Dh ~ 2*fin_gap
        """
        return 2 * self.fin_gap

    @property
    def flow_aspect_ratio(self):
        return self.fin_gap / self.fin_height


class Heatsink:

    def __init__(self, heatsink, fluid=air):
        self.heatsink = heatsink
        self.fluid = fluid

    def reynolds(self, velocity=1e-10):
        fluid = self.fluid
        heatsink = self.heatsink
        return fluid.density * heatsink.fin_gap ** 2 / fluid.viscosity / heatsink.length * velocity

    def nusselt(self, velocity=1e-10):
        fluid = self.fluid
        prandtl = fluid.prandtl
        reynolds = self.reynolds(velocity)
        return nusselt(reynolds, prandtl)

    def convection_coeff(self, velocity=1e-10):
        fluid = self.fluid
        heatsink = self.heatsink
        nusselt = self.nusselt(velocity)
        return nusselt * fluid.thermal_conductivity / heatsink.fin_gap

    def flow_rate(self, velocity=1e-10):
        return velocity * self.heatsink.flow_area

    def fin_efficiency(self, velocity=1e-10):
        heatsink = self.heatsink
        m = self.m(velocity)
        fin_height = heatsink.fin_height
        eff = np.tanh(m * fin_height) / (m * fin_height)
        return eff

    def tanhmL(self, velocity=1e-10):
        m = self.m(velocity)
        L = self.heatsink.fin_height
        return np.tanh(m*L)

    def m(self, velocity=1e-10):
        """
        m = sqrt(2h/kt) from Cengel's Intro to Thermodynamics and Heat Transfer, Chapter 10. For rectangular fins.
        """
        h = self.convection_coeff(velocity)
        heatsink = self.heatsink
        m = np.sqrt(2 * h / heatsink.fin_k / heatsink.fin_width)
        return m

    def thermal_resistance(self, velocity=1e-10):
        """
        https://www.electronics-cooling.com/2003/02/estimating-parallel-plate-fin-heat-sink-thermal-resistance/
        """
        h = self.convection_coeff(velocity)
        heatsink = self.heatsink
        base_area = heatsink.base_exposed_surface
        fin_area = heatsink.fin_exposed_surface
        fin_eff = self.fin_efficiency(velocity)
        fin_number = heatsink.fin_number
        conductivity = h * (base_area + fin_number * fin_eff * fin_area)
        return 1.0 / conductivity + heatsink.base_thermal_resistance

    def velocity_to_volume_flow(self, velocity):
        return self.heatsink.flow_area * velocity

    def volume_flow_to_velocity(self, flow):
        return flow / self.heatsink.flow_area

    def velocity_channel_to_approach(self, velocity):
        return velocity * (1 - self.heatsink.free_area_ratio)

    def velocity_approach_to_channel(self, velocity):
        return velocity / (1 - self.heatsink.free_area_ratio)

    def friction_factor_apparent(self, velocity):
        """
        velocity is channel velocity (inside fin gap).
        """
        fluid = self.fluid
        heatsink = self.heatsink

        reynolds = heatsink.hydraulic_diameter * velocity / fluid.viscosity_kinetic

        L_star = heatsink.length / heatsink.hydraulic_diameter / reynolds

        # friction factor
        aspect_ratio = heatsink.flow_aspect_ratio
        fRe_poly = np.poly1d(np.array([-6.089, 22.954, -40.829, 46.721, -32.527, 24]))
        fRe = fRe_poly(aspect_ratio)

        # apparent friction factor
        f_app = np.sqrt((3.44 / np.sqrt(L_star)) ** 2 + (fRe) ** 2) / reynolds
        return f_app

    def friction_factor_term(self, velocity):
        """
        velocity is channel velocity (inside fin gap).
        """
        f_app = self.friction_factor_apparent(velocity)
        fluid = self.fluid
        heatsink = self.heatsink
        ff_term = f_app * heatsink.fin_number * heatsink.length * (2 * heatsink.fin_height + heatsink.fin_gap) / \
                  heatsink.base_width / heatsink.fin_height
        return ff_term

    def contraction_term(self, velocity=None):
        return self.heatsink.flow_contraction_coeff

    def expansion_term(self, velocity=None):
        return self.heatsink.flow_expansion_coeff

    def pressure_drop_coeff(self, velocity):
        """
        velocity is channel velocity (inside fin gap).
        """
        fluid = self.fluid
        heatsink = self.heatsink
        Kc = heatsink.flow_contraction_coeff
        Ke = heatsink.flow_expansion_coeff
        ff_term = self.friction_factor_term(velocity)
        return Kc + Ke + ff_term

    def pressure_drop(self, velocity, correction_factor=2.0):
        """
        velocity is channel velocity (inside fin gap).
        """
        fluid = self.fluid
        K = (fluid.density * velocity ** 2) / 2 * correction_factor
        return self.pressure_drop_coeff(velocity) * K

def heatsink_from_config(config, fluid=air):
    hsb = HeatsinkBasic(base_width=config['base width'],
             base_height=config['base height'],
             base_k=config['base k'],
             fin_height=config['fin height'],
             fin_width=config['fin width'],
             fin_k=config['fin k'],
             fin_gap=config['fin gap'],
             length=config['length'])

    hs = Heatsink(heatsink=hsb, fluid=fluid)
    return hs