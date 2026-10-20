#Rocket simulation in order to learn how to use Python
#edoargenti


#Import all the modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci


##Constant parameters
G = 6.6742e-11 # Gravitational Constant (SI units)

#Planet (if you want to change the planet, you can put different Rplanet and Mplanet)

#EARTH
Rplanet = 6371000 #m
Mplanet = 5.972e24 #kg
name = 'Earth'
ATM_ALT = np.array([0, 11e3, 20e3, 32e3, 47e3, 51e3, 71e3, 86e3, 120e3])
ATM_RHO = np.array([1.225, 0.364, 0.0880, 0.0132, 0.00143,
                    0.00086, 6.4e-5, 6.96e-6, 2e-8])
OMEGA_EARTH = 7.2921159e-5
omega_vec = np.array([0.0, 0.0, OMEGA_EARTH])

#
x0 = Rplanet
y0 = 0
z0 = 0
v0 = np.cross(omega_vec, np.array([x0, y0, z0]))
velx0, vely0, velz0 = v0
#velx0 = 0
#vely0 = 0
#velz0 = 0
r0 = 200000 + Rplanet
period = 120000 #2*np.pi / np.sqrt(G*Mplanet) * r0**(3/2) * 1.5

#Rocket
stage1_dry_mass = 6_000 #kg
stage1_propellant = 65_300 #kg
stage2_dry_mass = 3_200 #kg
stage2_propellant = 24_500 #kg
payload_mass = 1_000 #kg
mass0 = (stage1_dry_mass + stage1_propellant + stage2_dry_mass + stage2_propellant + payload_mass) #kg
thrust1 = 1.5e6 #Newtons
thrust2 = 5e5 #Newtons
Isp1 = 300 #seconds
Isp2 = 345 #seconds
tMECO = 128 #Main Engine CutOff time
tsep1 = 2 #Lenght of time to remove the 1st stage
t2start = 130
t2end = t2start + 166
D = 0.85
A = np.pi*D**2 / 4
#CD = 0.1
 
"""
#KERBIN
Rplanet = 600000 #m
Mplanet = 5.2915158e22 #kg
name = 'Kerbin'

#Rocket
x0 = Rplanet
y0 = 0
z0 = 0
velx0 = 0
vely0 = 0
velz0 = 0
r0 = 200000 + Rplanet
period = 12000 #2*np.pi / np.sqrt(G*Mplanet) * r0**(3/2) * 1.5
weighttons = 5.3
mass0 = weighttons * 2000 / 2.2#kg
max_thrust = 167970 #Newtons
Isp1 = 250 #seconds
Isp2 = 400
tMECO = 38 #Main Engine CutOff time
tsep1 = 2 #Lenght of time to remove the 1st stage
mass1tons = 0.2
mass1 = mass1tons * 2000 / 2.2
t2start = 261
t2end = t2start + 20
D = 0.85
CD = 0.1
"""

#Speed of sound model
def speed_of_sound(altitude):
    if altitude < 11_000:
        temperature = 288.15 - 0.0065 * altitude
    elif altitude < 20_000:
        temperature = 216.65
    else:
        temperature = 216.65 + 0.001 * (altitude - 20_000)

    return np.sqrt(1.4 * 287.05 * temperature)

#Mach model
def drag_coefficient(mach):
    if mach < 0.8:
        return 0.25
    if mach < 1.2:
        return 0.25 + 0.35 * (mach - 0.8) / 0.4
    if mach < 5.0:
        return 0.60 - 0.20 * (mach - 1.2) / 3.8
    return 0.40


#Create a Aerodynamics class
class Aerodynamics():
    def __init__(self,name):
        self.name = name
        if name == 'Kerbin':
            #import the aero model for interpolation
            data = np.loadtxt('kerbin_aerodynamics.txt')
            self.altitude = data[:,0]
            self.density = data [:,3]
            self.rhos = self.density[0]
            self.beta = 0
        elif name == 'Earth' :
            #Earth aero model
            self.beta = 0.1354/1000 #density constant
            self.rhos = 1.225 #kg/m^3
            
    def getDensity(self,altitude):
        if self.name == 'Kerbin':
            #interpolation
            rho = np.interp(altitude, self.altitude, self.density)
        elif self.name == 'Earth':
            #use special equations
            #rho = self.rhos * np.exp(-self.beta*altitude) 
            #altitude = max(0.0, altitude)
            rho = np.exp(np.interp(altitude, ATM_ALT, np.log(ATM_RHO)))
        return rho


#Create the aeromodel variable
aeroModel = Aerodynamics(name)

##Gravitational Acceleration
def gravity(x,y,z):
    global Rplanet, Mplanet, G
    r = np.sqrt(x**2 + y**2 + z**2)
    
    if r < Rplanet:
        accelx = 0
        accely = 0
        accelz = 0
    else:
        accelx = G*Mplanet / (r**3) * x
        accely = G*Mplanet / (r**3) * y
        accelz = G*Mplanet / (r**3) * z
        
    return np.asarray([accelx,accely,accelz]),r

##Thrust
def propulsion(t,r_vec,v_vec):
    global Rplanet, tMECO, thrust1, Isp1, stage1_dry_mass, tsep1, t2start, t2end, thrust2, Isp2
    r_hat = r_vec/np.linalg.norm(r_vec) #local vertical
    axis = np.asarray([0,0,1]) #Earth rotation axis 
    east = np.cross(axis,r_hat)
    east = east / np.linalg.norm(east) #local east
    altitude = np.linalg.norm(r_vec) - Rplanet
    
    if t < tMECO:
        #1st stage: 10s completly vertical, then a gradual inclination toward east
        t_turn_start = 10
        t_turn_end = 120
        alpha_final = np.deg2rad(80)
        
        if t <= t_turn_start:
            alpha = 0
        else:
            fraction = (t - t_turn_start)/(t_turn_end - t_turn_start)
            fraction = np.clip(fraction, 0, 1)
            alpha = alpha_final * fraction
        
        direction = np.cos(alpha)*r_hat + np.sin(alpha)*east
        direction = direction/np.linalg.norm(direction)
        
        thrust = thrust1
        ve = Isp1 * 9.81
        mdot = -thrust/ve
    
    elif t < t2start:
        #Separation, no thrust
        direction = np.zeros(3)
        thrust = 0
        mdot = -stage1_dry_mass/tsep1
        
    elif t < t2end:
        #2nd stage
        #v_hat = v_vec / np.linalg.norm(v_vec)
        target_altitude = 150000
        if altitude < target_altitude:
            vertical_fraction = 0.1
        else:
            vertical_fraction = 0
        #vertical_fraction = np.clip((target_altitude-altitude)/target_altitude, 0, 0.15)
        #direction = (1-vertical_fraction)*v_hat + vertical_fraction*r_hat
        #direction = 0.7*east + 0.3*v_hat
        direction = east + vertical_fraction*r_hat
        direction = direction/np.linalg.norm(direction)
        
        thrust = thrust2
        ve = Isp2 * 9.81
        mdot = -thrust/ve
        
    else:
        direction = np.zeros(3)
        thrust = 0
        mdot = 0
        
    thrustF = thrust * direction
    
    return thrustF, mdot
        
    

"""
def propulsion(t):
    #Timing for thrusters
    global thrust1, thrust2, Isp1, Isp2, tMECO
    if t < tMECO:
        #the main thruster is firing
        direction = np.asarray([1,0,0])
        #theta = 10 * np.pi / 180
        thrustF = thrust1
        ve = Isp1 * 9.81 #m/s
        #mdot 
        mdot = -thrustF/ve
    if t > tMECO and t < (tMECO + tsep1):
        direction = np.asarray([0,0,0])
        #theta = 0
        thrustF = 0
        #masslost = mass1
        mdot = -mass1/tsep1
    if t > (tMECO + tsep1):
        direction = np.asarray([0,0,0])
        #theta = 0
        thrustF = 0 
        mdot = 0
    if t > t2start and t < t2end:
        direction = np.asarray([0,1,0])
        #theta = 90 * np.pi / 180
        thrustF = thrust2
        ve = Isp2 * 9.81 #m/s
        mdot = -thrustF/ve
    if t > t2end:
        direction = np.asarray([0,0,0])
        #theta = 0
        thrustF =  0
        mdot = 0
        
    #theta = 0 #Angle of the thrust
    #thrustx = thrustF*np.cos(theta)
    #thrusty = thrustF*np.
    #thrustz = thrustF*np.sin(theta)
    
    thrustF = thrustF*direction
    
    return thrustF, mdot
"""
##Equations of motion
# F = m * a = m * zddot
# x is the altitude from the center of the planet along the equator [m]
# z is the altitude from the center of the planet along the north pole [m]
# xdot is the velocity along x
# zdot is the velocity along z
# xddot is the acceleration along x
# zddot is the acceleration along z
##Second Order Derivatives
def Derivatives(state,t):
    global aeroModel, Rplanet
    #state vector
    x = state[0]
    y = state[1]
    z = state[2]
    velx = state[3]
    vely = state[4]
    velz = state[5]
    mass = state[6]
    
    #Compute zdot - kinematic Relationship
    xdot = velx
    ydot = vely
    zdot = velz
    
    #Rotation of the Earth
    r_vec = np.array([x, y, z])
    v_vec = np.array([velx, vely, velz])

    v_atmosphere = np.cross(omega_vec, r_vec)
    v_rel = v_vec - v_atmosphere
    
    #Compute the total forces
    #Gravity
    accel,r = gravity(x,y,z) 
    gravityF = -accel * mass
    
    #Aerodynamics
    altitude = r - Rplanet #altitude above surface
    rho = aeroModel.getDensity(altitude) #Air Density
    speed_rel = np.linalg.norm(v_rel)
    mach = speed_rel / speed_of_sound(altitude)
    CD = drag_coefficient(mach)

    if speed_rel > 1e-9:
            aeroF = -0.5 * rho * CD * A * speed_rel * v_rel
    else:
            aeroF = np.zeros(3)
            
    #altitude = r - Rplanet #altitude above surface
    #rho = aeroModel.getDensity(altitude) #Air Density
    #V = np.sqrt(v_rel[0]**2 + v_rel[1]**2 + v_rel[2]**2)
    #qinf = (np.pi/8) * rho * D**2 * abs(V) #Dynimic pressure
    #aeroF = -qinf*CD*np.asarray([velx,vely,velz])
    
    #Thrust
    r_vec = np.asarray([x,y,z])
    v_vec = np.asarray([velx,vely,velz])
    thrustF,mdot = propulsion(t,r_vec,v_vec)
    
    #Total Forces
    Forces = gravityF + aeroF + thrustF
    
    #Compute the acceleration
    if mass > 0:
        ddot = Forces/mass
    else:
        ddot = 0
        mdot = 0
    
    #Compute the statedot
    statedot = np.asarray([xdot,ydot,zdot,ddot[0],ddot[1],ddot[2],mdot])
    
    return statedot

##Main Script

#Test Gravity
accel,r = gravity(0,0,Rplanet) 
print('Surface Gravity(m/s^2) = ', accel[0], accel[1], accel[2])

#Plot the air density as a function Above Ground Level (AGL)
test_altitude = np.linspace(0,100000,100)
test_rho = aeroModel.getDensity(test_altitude)
plt.figure()
plt.plot(test_altitude,test_rho,'-b')
plt.xlabel('Altitude (m)')
plt.ylabel('Air Density (kg/m^3)')
plt.grid()


#Initial conditions - for orbit
"""
x0 = Rplanet + 600000
z0 = 0
r0 = np.sqrt(x0**2 + z0**2)
velx0 = 0 #m/s 
velz0 = np.sqrt(G*Mplanet / r0) * 1.1 #m/s
period = 2*np.pi / np.sqrt(G*Mplanet) * r0**(3/2) * 1.5
"""

#Initial conditions - for single stage rocket
stateinitial = np.asarray([x0,y0,z0,velx0,vely0,velz0,mass0]) 

#Time window
tout = np.linspace(0,period,1000)

#Numerical integration call
stateout = sci.odeint(Derivatives, stateinitial, tout)

#Rename variables
xout = stateout[:,0]
yout = stateout[:,1]
zout = stateout[:,2]
altitude = np.sqrt(xout**2 + yout**2 + zout**2) - Rplanet
velxout = stateout[:,3]
velyout = stateout[:,4]
velzout = stateout[:,5]
velout = np.sqrt(velxout**2 + velyout**2 + velzout**2)
massout = stateout[:,6]

#Keplerian elements
i = np.argmin(np.abs(tout - t2end))

r_vec = np.array([xout[i], yout[i], zout[i]])
v_vec = np.array([velxout[i], velyout[i], velzout[i]])

r = np.linalg.norm(r_vec)
v = np.linalg.norm(v_vec)

r_hat = r_vec / r
v_radial = np.dot(v_vec, r_hat)
v_tangenzial = np.sqrt(v**2 - v_radial**2)

mu = G * Mplanet
E = v**2 / 2 - mu / r
h = np.linalg.norm(np.cross(r_vec, v_vec))
e = np.sqrt(1 + 2 * E * h**2 / mu**2)
a = -mu / (2 * E)

rp = a * (1 - e)
ra = a * (1 + e)

print("Burnout Altitude [km]:", (r - Rplanet) / 1000)
print("Total Velocity [km/s]:", v / 1000)
print("Radial Velocity [m/s]:", v_radial)
print("Tangential Velocity [km/s]:", v_tangenzial / 1000)
print("Perigee [km]:", (rp - Rplanet) / 1000)
print("Apogee [km]:", (ra - Rplanet) / 1000)

##PLOTS

#Plot Altitude
plt.figure()
plt.plot(tout, altitude)
plt.xlabel('Time (sec)')
plt.ylabel('Altitude (m)')
plt.grid()

#Plot Velocity 
plt.figure()
plt.plot(tout,velout)
plt.xlabel('Time (sec)')
plt.ylabel('Total Speed (m/s)')
plt.grid()

#Mass
plt.figure()
plt.plot(tout,massout)
plt.xlabel('Time (sec)')
plt.ylabel('Mass (kg)')
plt.grid()

"""
#2D Orbit
plt.figure()
plt.plot(xout,zout,'-r',label='Orbit')
plt.plot(xout[0],zout[0],'g*')
theta = np.linspace(0,2*np.pi,1000)
xplanet = Rplanet * np.sin(theta)
yplanet = Rplanet * np.cos(theta)
plt.plot(xplanet,yplanet,'-b',label='Planet')
plt.grid()
plt.legend()
"""

#3D orbit
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(xout,yout,zout,'-r', label='Trajectory')
ax.scatter(xout[0],yout[0],zout[0],'*g',label='Launch')

u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 40)
X = Rplanet * np.outer(np.cos(u), np.sin(v))
Y = Rplanet * np.outer(np.sin(u), np.sin(v))
Z = Rplanet * np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(X,Y,Z, color='b', alpha=0.35)

lim = max(np.max(np.abs(xout)), np.max(np.abs(yout)), np.max(np.abs(zout)), Rplanet)
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_zlim(-lim,lim)
ax.set_box_aspect((1,1,1))

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()
plt.show()







