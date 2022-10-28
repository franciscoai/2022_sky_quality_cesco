import numpy as np

def pressure_water(T):
	exponent = (18.678 - T/234.5)*(T/(257.14 + T))	
	es = 611.21*np.exp(exponent)      # The partial pressure of water vapor (Pa)
	return es

def water_vapor(T, RH):
    # Purpose: 
    # Estimate the Precipitable Water Vapor (PWV) in mm from local wheather measurements
    # Based in the work by Otarola et al. (2010) 
    # param: T: Temperature (°C, e.g. T = 20)
    # param: RH: Relative Humedity (%, e.g. RH = 25)

    # Constants:
    H = 1.5					# the PWV Height scale in km
    z0 = 2.370				# surface altitude in km
    zmax = 12  				# maximum altitude for PWV contribution in km (zmax = 12 km from Otarola et al. (2010))
    Rv = 461.9 				# The water vapor gas constant (J Kg^-1 K^-1)

    TK = T + 273.15         # Temperature in Kelvin 
    es = pressure_water(T)
    e0 = es*(RH/1E+2)  			# the surface level partial vapor pressure (Pa)
    rho_v0 = e0/(Rv*TK)			# the surface water vapor density (Kg m^-3)
    PWV = rho_v0*H*1E+3*(1 - np.exp((z0-zmax)/H))
    return PWV # Precipitable Water Vapor (PWV) in mm 
