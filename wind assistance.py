import numpy as np 
import math


V_wind = 40 #m/s  wind speed
theta = 20 #deg   wind direction
C = 9 #TBD
scaling = 1/1000 #use same as the wind 
def get_assistance_force(wind_speed, wind_direction_deg):
    force_magnitude = C * (wind_speed ** 2)
    

    radians = math.radians(wind_direction_deg)
    
    wind_x = force_magnitude * math.cos(radians)
    wind_y = force_magnitude * math.sin(radians)
    

    F_assist_x = -wind_x
    F_assist_y = -wind_y
    
    return F_assist_x, F_assist_y

f_ax, f_ay = get_assistance_force(V_wind, theta)

total_assist = np.sqrt(f_ax**2 + f_ay**2)
scaled_assist = total_assist*scaling
print(f"Assistance force X: {f_ax:.2f}N, Y: {f_ay:.2f}N")
print(f"Totale kracht: {total_assist:.2f}N")

if scaled_assist > 4:
    print (f"Assisting with: {scaled_assist:.2f}N ")
    
else:
    print (f"Not assisting force too small: {scaled_assist:.2f}N ")
    