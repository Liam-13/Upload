from chroma.event import Photons
from chroma.sample import uniform_sphere
from chroma.transform import rotate
import numpy as np
import time

def UseGenerator(NPhotons, Position, Direction=None, Wavelength=178.0, Generator='PhotonBomb',Diameter=1.0, ExtraData=None): 
    if Generator == 'PhotonBomb': 
        Photons = PhotonBomb(NPhotons, Wavelength, Position)
    elif Generator == 'PhotonAreaBomb': 
        Photons = PhotonAreaBomb(NPhotons, Wavelength, Position, Diameter)
    elif 'Laser' in Generator:
        Diameter = float(Generator.split(',')[1])
        Divergence = float(Generator.split(',')[2])
        Photons = Laser(NPhotons, Wavelength, Position, Direction, Diameter, Divergence)
    elif 'Flashlight' in Generator: 
        Angle = float(Generator.split(',')[1])/180.0 * np.pi
        Photons = Flashlight(NPhotons, Wavelength, Position, Direction, Angle)
    elif 'Beam' in Generator:
        if ',' in Generator:
            Direction = Generator.split(',')[1]
            Direction = GetAxis(Direction)
        Photons = Beam(NPhotons, Wavelength, Position, Direction)
    elif 'CherenkovSimple' in Generator:
        if ';' in Generator:
            Momentum = Generator.split(';')[1]
            Momentum = Momentum.split('[')[1].split(']')[0]
            Momentum = np.array([float(i) for i in Momentum.split(',')])#get x,y,z coords in an array
            Length = float(Generator.split(';')[2]) #length in mm

        Energy = 1 #GeV
        Photons = CherenkovSimple(NPhotons, Position, Momentum, Length, Energy)
    elif 'MuonInput' in Generator:
        '''
        Takes the data read from a csv and gets the values needed for 
        CherenkovSimple(), one photon at a time (so ExtraData should only contain
        data for one photon each time this function is called).

        Might want to stop photons being generated in the OC later from muons 
        passing through the OC
        '''
        Energy = ExtraData[3][0] #GeV
        #take length of vector between the exit and entry points
        Length = np.linalg.norm(ExtraData[1] - ExtraData[0]) #mm
        
        UnitDirection = (ExtraData[1] - ExtraData[0]) / Length
        Momentum = UnitDirection

        print(Position, Momentum, Length, Energy)
        Photons = CherenkovSimple(NPhotons, Position, Momentum, Length, Energy)

    return Photons 

def PhotonBomb(NPhotons,Wavelength,Position):
    Position = np.tile(Position, (NPhotons,1))
    Direction = uniform_sphere(NPhotons)
    Polarization = np.cross(Direction, uniform_sphere(NPhotons))
    Wavelengths = np.repeat(Wavelength, NPhotons)
    return Photons(Position, Direction, Polarization, Wavelengths)

def PhotonAreaBomb(NPhotons, Wavelength, Position, Diameter):
    radii = np.random.uniform(0, Diameter/2, NPhotons)
    angles = np.random.uniform(0, 2*np.pi, NPhotons)
    points = np.empty((NPhotons,3))

    points[:,0] = np.sqrt(Diameter/2)*np.sqrt(radii)*np.cos(angles) + Position[:,0]
    points[:,1] = np.repeat(Position[:,1], NPhotons)
    points[:,2] = np.sqrt(Diameter/2)*np.sqrt(radii)*np.sin(angles) + Position[:,2]
    
    #Use flipped direction for ETS - source should face in z direction as detectors 
    #are on flat surface of cyclinder, not the walls like nEXO
    #points[:,0] = np.sqrt(Diameter/2)*np.sqrt(radii)*np.cos(angles) + Position[0]
    #points[:,1] = np.sqrt(Diameter/2)*np.sqrt(radii)*np.sin(angles) + Position[1]
    #points[:,2] = np.repeat(Position[2], NPhotons)
    
    Position = points
    # print(Position)
    Direction = uniform_sphere(NPhotons)
    Polarization = np.cross(Direction, uniform_sphere(NPhotons))
    # print(Direction)
    # print(uniform_sphere(NPhotons))
    Wavelengths = np.repeat(Wavelength, NPhotons)
    return Photons(Position, Direction, Polarization, Wavelengths)

def Beam(NPhotons, Wavelength, Position, Direction):
    Position = np.tile(Position, (NPhotons,1))
    Direction = np.tile(Direction, (NPhotons,1))
    Polarization = np.cross(Direction,uniform_sphere(NPhotons))
    Wavelengths = np.repeat(Wavelength,NPhotons)
    return Photons(Position,Direction,Polarization,Wavelengths)

def Laser(NPhotons, Wavelength, Position, Direction, Diameter, Divergence):
    # normalize the direction vector
    d = np.sqrt(np.sum([ii ** 2 for ii in Direction]))
    t = np.arctan2(Direction[1],Direction[0])
    p = np.arccos(Direction[2]/d)
    Direction = (np.sin(p)*np.cos(t),np.sin(p)*np.sin(t),np.cos(p)) # np.around((np.sin(p)*np.cos(t),np.sin(p)*np.sin(t),np.cos(p)),6)
    
    maxAngle = Divergence/2 
    deltaZ = -(Diameter/2)/np.tan(maxAngle)
    
    # mean = [Position[0],Position[1]]
    mean = [0,0] #always keep it centered, rotate and stuff first, then shift
    sigma = np.array([Diameter/4,Diameter/4])
    cov = np.diag(sigma**2)

    # for polarization
    X,Y = np.random.multivariate_normal(mean,cov,NPhotons).T
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y,X)
    Phi = np.arctan(R/deltaZ)
    
    # For actual location on disk
    x,y = np.random.multivariate_normal(mean,cov,NPhotons).T
    radii = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    phi = np.arctan(radii/deltaZ)
    points = np.empty((NPhotons,3))
    points[:,0] = x 
    points[:,1] = -y 
    points[:,2] = np.repeat(0,NPhotons)

    d = Direction
    
    rotation_angle, rotation_axis = findRotParams(Direction)

    Position = Position + rotate(points, rotation_angle, rotation_axis)
    Direction = laser(phi=phi, theta=theta, direction=Direction, size=NPhotons) # , np.pi, Direction)
    Polarization = np.cross(Direction, laser(phi=Phi, theta=Theta, direction=d, size=NPhotons))
    Wavelengths = np.repeat(Wavelength, NPhotons)
    return Photons(Position, Direction, Polarization, Wavelengths)

def laser(phi=None, theta=None, direction=(0,0,-1), size=None, dtype=np.double):
    u = abs(np.cos(phi))

    c = np.sqrt(1-u**2)

    rotation_angle, rotation_axis = findRotParams(direction)

    if size is None:
        return rotate(np.array([c*np.cos(theta), c*np.sin(theta), u]),
                      rotation_angle, rotation_axis)

    points = np.empty((size, 3), dtype)
    points[:,0] = c*np.cos(theta)
    points[:,1] = -c*np.sin(theta)
    points[:,2] = u

    return rotate(points, rotation_angle, rotation_axis)

def findRotParams(direction):
    # print(direction)
    if np.equal(direction, (0,0,1)).all():
        rotation_axis = (0,0,1)
        rotation_angle = 0.0
    elif np.equal(direction, (0,0,-1)).all():
        rotation_axis = (1,0,0)
        rotation_angle = np.pi
    else:
        rotation_axis = np.cross((0,0,1), direction)
        rotation_angle = \
                -np.arccos(np.dot(direction, (0,0,1))/np.linalg.norm(direction))

    return rotation_angle, rotation_axis

def Flashlight(NPhotons, Wavelength, Position, Direction, Angle):
    Position = np.tile(Position, (NPhotons,1))
    Direction = flashlight(phi=Angle, direction=Direction, size=NPhotons) 
    Polarization = np.cross(Direction, flashlight(size=NPhotons))
    Wavelengths = np.repeat(Wavelength,NPhotons)
    return Photons(Position, Direction, Polarization, Wavelengths)

def flashlight(phi=np.pi/4, direction=(0,0,1), size=None, dtype=np.double):
    theta = np.random.uniform(0.0, 2*np.pi, size)
    u = np.random.uniform(np.cos(phi), 1, size)
    c = np.sqrt(1-u**2)

    if np.equal(direction, (0,0,1)).all():
        rotation_axis = (0,0,1)
        rotation_angle = 0.0
    elif np.equal(direction, (0,0,-1)).all():
        rotation_axis = (1,0,0)
        rotation_angle = np.pi
    else:
        rotation_axis = np.cross((0,0,1), direction)
        rotation_angle = \
            -np.arccos(np.dot(direction, (0,0,1))/np.linalg.norm(direction))

    if size is None:
        return rotate(np.array([c*np.cos(theta), c*np.sin(theta), u]),
                      rotation_angle, rotation_axis)

    points = np.empty((size, 3), dtype)
    points[:,0] = c*np.cos(theta)
    points[:,1] = c*np.sin(theta)
    points[:,2] = u

    return rotate(points, rotation_angle, rotation_axis)

def CherenkovSimple(NPhotons, Position, Momentum, Length, Energy):
    """
    This does all the real work for generating simple Cherenkov light, imitating the
    light that a muon traveling with the momentum and initial position given in the yaml
    file for the given length. This will create Length*photonsPerMm number of source, all
    of which will only generate one photon with a pseudo random direction at the 42 deg
    cherenkov angle in water relative to the orginal 'muon track'.
    """
    ### Get beta from muon energy###
    #using 1 GeV for muon energy for now, but really this will come from the muon input data
    MuonEnergy = 10 #GeV
    gamma = MuonEnergy/0.10566 #divide by the muon mass [0.10566 GeV]
    beta =  np.sqrt(1.0 - 1.0/(gamma**2.0)) #should be [.9, .99]
    n = 1.33 #approximate index of refraction of water

    parentMomentum = np.array([Momentum[0], Momentum[1], Momentum[2]])

    parentMomentumUnit = parentMomentum/np.linalg.norm(parentMomentum) 
    #so we assume the distance is set by the Length, and should use parentMomentumUnit
    #to calculate any distances traveled
    
    parentEntryPosition =  np.array([Position[0], Position[1], Position[2]])
    
    parentEndPosition = Length*parentMomentumUnit + parentEntryPosition
    
    #getting energy dependant number of photons per length by integrating over the wavelengths we
    #care about and using beta, using the Frank-Tamm formula
    alpha = 1/137. #fine structure constant
    nIndex, WaterWavelength = GetWaterProperties() #get array of wavelength we will use

    #using the min/max wavelengths, get the number of photons per unit length
    #lower bound may be too low right now
    N = 2*np.pi*alpha*(1/np.min(WaterWavelength) - 1/np.max(WaterWavelength) )*(1-1/((beta*n)**2))
    photonsPerMm = N * 1e6 #convert from nm to mm

    #radians, Cherenkov angle dependant on muon energy
    thetaRel = np.arccos(1/(beta*n))
    
    CNumPhotons = int(Length*photonsPerMm)
    if CNumPhotons != NPhotons:
        print("{} photons were requested but for Cherenkov photons for a length of {} mm, {} photons " \
                "will be used.".format(NPhotons,Length,CNumPhotons))
                #"should really be expected.".format(NPhotons,Length,CNumPhotons))
    #rewrites how many photons it will produce
    NPhotons = CNumPhotons
    
    #set up arrays to hold our photon positions and directions
    photonsPosition = np.zeros((NPhotons, 3))
    photonsDirection = np.zeros((NPhotons, 3))
    
    #start_time = time.time() #for timing how long this takes
    
    numStepsAway = np.random.rand(NPhotons)*Length #pick a random point on the track

    #photon position
    photonsPosition[:,0] = parentEntryPosition[0]+numStepsAway*parentMomentumUnit[0]
    photonsPosition[:,1] = parentEntryPosition[1]+numStepsAway*parentMomentumUnit[1]
    photonsPosition[:,2] = parentEntryPosition[2]+numStepsAway*parentMomentumUnit[2]

    #now photon direction (aka momentum)
    #First, rotate relative to z axis by Cherenkov angle
    c, s = np.cos(thetaRel), np.sin(thetaRel) #cos and sine

    #this finds a vector perpendicular to the parentMomentum one (unless parentMomentum==[0.,0.,1.0])
    ux, uy, uz = get_perp(parentMomentumUnit)

    R = np.array([[c+ux*ux*(1.0-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s], [uy*ux*(1-c)+uz*s, c+uy*uy *(1-c), \
            uy*uz*(1-c)-ux*s], [uz*ux*(1-c) - uy*s, uz*uy*(1-c)+ux*s, c+uz*uz * (1-c) ]])#rotation matrix
    
    momentum = R.dot(parentMomentum)

    #Now rotate random amount of phi about axis of parent trajectory
    phi=np.random.sample(NPhotons)*np.pi*2.0
    c, s = np.cos(phi), np.sin(phi) #cos and sine
    ux, uy, uz = parentMomentumUnit
    R = np.array([[c+ux*ux*(1.0-c), ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s], [uy*ux*(1-c)+uz*s, c+uy*uy *(1-c), uy*uz*(1-c)-ux*s], [uz*ux*(1-c) - uy*s, uz*uy*(1-c)+ux*s, c+uz*uz * (1-c) ]])#rotation matrix
    
    #need to transpose to do the dot product along the correct axis
    R = np.transpose( R.T, axes=(0,2,1))
    momentum = np.dot(R, momentum)

    #Photon momentum set so now leave track in a cherenkov cone
    #which also gives us the photons' direction
    norm = np.linalg.norm(momentum, axis=1)
    Direction = momentum/norm.reshape((norm.size, 1)) #get unit direction 

    Position = photonsPosition
    
    # get photon wavelengths from sampling from an intensity distribution
    Wavelengths = getCherenkovWavelengthSample(NPhotons, beta)
    Polarization = np.cross(Direction,uniform_sphere(NPhotons))
    
    #print("Time:", time.time() - start_time, "to end of gen.")
    return Photons(Position,Direction,Polarization,Wavelengths)

def getCherenkovWavelengthSample(NumPhotons, beta):  
    '''
    Returns NumPhotons number wavelengths [nm] sampled from a wavelength vs intensity distribution
    calculated from the Frank-Tamm Formula.
    '''

    nIndex, wavelengths = GetWaterProperties() 
    #get frequencies from the energy array
    freq = 3e17/wavelengths
    
    #calculate the energy lost per unit length from the Frank-Tamm Formula
    #assuming mu(w) is constant in this range
    E_loss = 1.0/(4*np.pi) * freq*(1.0 - 1.0/(beta**2.0 * nIndex**2.0))
    E_loss_rel = E_loss/np.abs(E_loss.sum()) #renormalize
    
    ### Interpolating the data to get a smooth sampling ###
    wavelength_intep = np.linspace(np.min(wavelengths), np.max(wavelengths), 300)
    #need to flip data arrays as np.interp needs increasing x values
    E_loss_rel_intep = np.interp(wavelength_intep, wavelengths[::-1], E_loss_rel[::-1])
    #normalize
    E_loss_rel_intep = E_loss_rel_intep/np.sum(E_loss_rel_intep)
    
    #take the sample using the intensity as the weights
    photonWavelen = np.random.choice(wavelength_intep, p=E_loss_rel_intep, size=NumPhotons)
    
    return photonWavelen

def GetWaterProperties():

    #From nEXO_offline
    energies = np.array([2.07, 2.09, 2.12, 2.20, 2.31, 2.41, 2.48, 2.55, 2.71, 2.83 ,
           2.96, 3.05, 3.13, 3.25, 3.32, 3.45, 3.57, 3.69, 3.85, 3.99, 4.26, 4.99, 5.29]) #eV

    nIndex = np.array([1.332, 1.33233458, 1.33275974, 1.333, 1.33337057,1.33440426, 1.33497163,
            1.33553901, 1.33667376, 1.33746809, 1.33824548, 1.33866779, 1.33926229, 1.34044132,
            1.34115378, 1.3422317 , 1.34324954, 1.34459331, 1.34635211, 1.34761808, 1.35056125,
            1.36267944, 1.36844391])

    wavelengths = 1239.8/energies #converts from energy in eV to wavelength in nm
    return nIndex, wavelengths

def get_perp(x):
    """Returns an arbitrary vector perpendicular to `x`."""
    a = np.zeros(3)
    a[np.argmin(abs(x))] = 1
    return np.cross(a,x)

def GetAxis(Axis):
    if '-' in Axis: 
        Num = -1
    else: 
        Num = 1 
    if 'X' in Axis: 
        Vector = (Num,0,0)
    elif 'Y' in Axis: 
        Vector = (0,Num,0)
    elif 'Z' in Axis: 
        Vector = (0,0,Num)
    return Vector
