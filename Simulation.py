import chroma, random, datetime, os, sys, time, scipy.spatial

import Photons, Output
from Utilities import *
from MuonGeneration2 import MuonGen, ReadMuonsFromCSV2
import PlotFunctions as Plot

import numpy as np
import matplotlib.pyplot as plt 

class Simulation(object):
    def __init__(self, Detector):
        self.Detector = Detector
        self.Yaml = self.Detector.Yaml
        self.OpticalParameters = self.Detector.OpticalParameters
        self.DetectorPos = self.Detector.GeoData['DetectorPos']
        self.PhotonEvents = []
        self.RecordIncidentAngles = False
        self.Triangles = self.Detector.GeoData['TotalMesh'].assemble()
        self.PhotonTags = ['No Hit','Bulk Absorption','Surface Detection',
                           'Surface Absorption','Rayleigh Scattering',
                           'Diffuse Reflection','Specular Reflection',
                           'Surface Reemission','Surface Transmission',
                           'Bulk Reemission','Material Reflection','NAN']
    
    def GeneratePhotons(self):
        ExtraData=None #extra value to pass to UseGenerator to pass additional data needed from the yaml file

        # print(self.Yaml['Simulation']['PhotonLocation'])
        if 'Uniform' in self.Yaml['Simulation']['PhotonLocation']: 
            if self.Yaml['Simulation']['PhotonLocation'] == 'UniformFV':     
                MinZ = self.Detector.GeoData['FiducialHeight'][0]
                MaxZ = self.Detector.GeoData['FiducialHeight'][1]
                MinR = 0.0 
                MaxR = self.Detector.GeoData['FiducialRadius']
            elif self.Yaml['Simulation']['PhotonLocation'] == 'UniformXenon':
                MinZ = self.Detector.GeoData['SkinHeight'][0]
                MaxZ = self.Detector.GeoData['SkinHeight'][1]
                MinR = 0.0 
                MaxR = self.Detector.GeoData['SkinRadius'][1]
            elif self.Yaml['Simulation']['PhotonLocation'] == 'UniformSkin':
                MinZ = self.Detector.GeoData['SkinHeight'][0]
                MaxZ = self.Detector.GeoData['SkinHeight'][1]
                MinR = self.Detector.GeoData['SkinRadius'][0]
                MaxR = self.Detector.GeoData['SkinRadius'][1]
            Theta = np.random.uniform(0, 2.0*np.pi, self.Yaml['Simulation']['NumberOfSources'])
            Radius = np.sqrt(np.random.uniform(MinR**2, MaxR**2, self.Yaml['Simulation']['NumberOfSources']))
            X = Radius * np.cos(Theta)
            Y = Radius * np.sin(Theta)
            Z = np.random.uniform(MinZ, MaxZ, self.Yaml['Simulation']['NumberOfSources'])
            Points = np.stack((X,Y,Z), axis=-1)
            Direction = [None]*self.Yaml['Simulation']['NumberOfSources']
        elif 'Z=' in self.Yaml['Simulation']['PhotonLocation']:
            Points = []
            Z = self.Yaml['Simulation']['PhotonLocation'].split('=')[1]
            if ',' in Z:
                Z = np.array(Z.split(','), dtype=np.float)
            else:
                print('The correct format for simulating between two radii is Z=Z1,Z2')
                sys.exit()
            MinR = 0
            MaxR = self.Detector.GeoData['SkinRadius'][1]
            Theta = np.random.uniform(0, 2.0*np.pi, self.Yaml['Simulation']['NumberOfSources'])
            Radius = np.sqrt(np.random.uniform(MinR**2, MaxR**2, self.Yaml['Simulation']['NumberOfSources']))
            X = Radius * np.cos(Theta)
            Y = Radius * np.sin(Theta)
            Z = np.random.uniform(Z[0], Z[1], self.Yaml['Simulation']['NumberOfSources'])
            Points = np.stack((X,Y,Z), axis=-1)
            Direction = [None]*self.Yaml['Simulation']['NumberOfSources']
        elif 'R=' in self.Yaml['Simulation']['PhotonLocation']: 
            Points = []
            R = self.Yaml['Simulation']['PhotonLocation'].split('=')[1]
            if ',' in R: 
                R = np.array(R.split(','), dtype=np.float)
            else: 
                print('The correct format for simulating between two radii is R=R1,R2')
                sys.exit()
            MinZ = self.Detector.GeoData['SkinHeight'][0]
            MaxZ = self.Detector.GeoData['SkinHeight'][1]
            Theta = np.random.uniform(0, 2.0*np.pi, self.Yaml['Simulation']['NumberOfSources'])
            Radius = np.sqrt(np.random.uniform(R[0]**2, R[1]**2, self.Yaml['Simulation']['NumberOfSources']))
            X = Radius * np.cos(Theta)
            Y = Radius * np.sin(Theta)
            Z = np.random.uniform(MinZ, MaxZ, self.Yaml['Simulation']['NumberOfSources'])
            Points = np.stack((X,Y,Z), axis=-1)
            Direction = [None]*self.Yaml['Simulation']['NumberOfSources']
        elif self.Yaml['Simulation']['PhotonLocation'] == 'Center':
            TopMesh = self.Yaml['Detector']['FiducialVolume']['Top']
            BottomMesh = self.Yaml['Detector']['FiducialVolume']['Bottom']
            Top = self.Detector.GetMaxHeight(np.sum(self.Detector.GeoData['Meshes'][TopMesh]))
            Bottom = self.Detector.GetMinHeight(np.sum(self.Detector.GeoData['Meshes'][BottomMesh]))
            MidHeight = Top - (np.abs(Top-Bottom))/2.0
            Center = np.array([0.0, 0.0, MidHeight])
            Points = np.tile(Center, (self.Yaml['Simulation']['NumberOfSources'],1))     
            Direction = [None]*self.Yaml['Simulation']['NumberOfSources']
        elif isinstance(self.Yaml['Simulation']['PhotonLocation'], list): 
            Points = np.tile(np.array(self.Yaml['Simulation']['PhotonLocation']), (self.Yaml['Simulation']['NumberOfSources'],1))
            Direction = [None]*self.Yaml['Simulation']['NumberOfSources']
        elif 'Area=' in self.Yaml['Simulation']['PhotonLocation']:
            Points = []
            Circle = self.Yaml['Simulation']['PhotonLocation'].split('=')[1]
            if ',' in Circle:
                Circle = np.array(Circle.split(','), dtype=np.float)
            else:
                print('The correct format for simulating from a circular cross-section is Area= x,y,z,r')
                sys.exit()
            Theta = np.random.uniform(0, 2.0*np.pi, self.Yaml['Simulation']['NumberOfSources'])
            Radius = np.random.uniform(0, Circle[3]**2, self.Yaml['Simulation']['NumberOfSources'])
            X = Circle[0] + np.sqrt(Radius) * np.cos(Theta)
            Y = Circle[1] + np.sqrt(Radius) * np.sin(Theta)
            Z = np.full(self.Yaml['Simulation']['NumberOfSources'],Circle[2])
            Points = np.stack((X,Y,Z), axis=-1)
            print("Source locations: ", Points)
            Direction = [None]*self.Yaml['Simulation']['NumberOfSources']

        # Simulate photons according to one of two laser calibration schemes:
        # Scheme 1 = LaserCalibrationCenter: Photons are emitted from 4 different fibers 
        # which are located at the center of the TPC between: two SiPM Staves. 
        # Photons in this configuration will be emitted towards the center of the TPC
        elif self.Yaml['Simulation']['PhotonLocation'] == 'LaserCalibrationCenter':
            # Get center location of all staves and field rings 
            CentersStaves = np.array([self.Detector.GetCenter(x) for x in self.Detector.GeoData['Meshes']['Backplane']])
            CentersRings = np.array([self.Detector.GetCenter(x) for x in self.Detector.GeoData['Meshes']['Field-Shaping-Rings']])

            # Get spacing between field rings and check that Z-position of laser fiber is not between same as field rings 
            Spacing = np.mean(np.abs(np.diff(CentersRings[:,2])))
            Offset = np.mean(CentersStaves[:,2]) - CentersRings[:,2]

            # If laser fiber is too close to field ring, add offset to place it between field rings
            DiffZ = 0.0
            if np.min(np.abs(Offset)) < Spacing/2.0: 
                Cut = np.where(np.abs(Offset) == np.min(np.abs(Offset)))[0]
                if Offset[Cut] < 0.0: 
                    DiffZ = -(Spacing/2.0 + Offset[Cut][0])
                else: 
                    DiffZ = Spacing/2.0 - Offset[Cut][0]

            # Calculate angle and radius of above 3D coordinates
            Radius = [np.sqrt(x[0]**2 + x[1]**2) for x in CentersStaves]
            Phi = [np.arctan2(x[1], x[0]) for x in CentersStaves]

            # Take angular difference between consecutive staves 
            # to calculate angular position of gaps between staves
            DiffPhi = np.abs(np.diff(Phi)[0])
            PhiFiber = Phi[::int(len(Phi)/self.Yaml['Simulation']['NumberOfSources'])]

            # Transform polar coordinates back into cartesian coordinates 
            # add small angle offset of 0.2 rad to be away form the Sapphire Rods 
            X = np.mean(Radius)*0.95 * np.cos(PhiFiber+DiffPhi/2.0+0.2)
            Y = np.mean(Radius)*0.95 * np.sin(PhiFiber+DiffPhi/2.0+0.2)
            Z = np.array([np.mean(CentersStaves[:,2])+DiffZ]*len(X))
            Points = np.dstack((X,Y,Z))[0]

            Direction = np.array([0.0, 0.0, np.mean(CentersStaves[:,2])]) - Points
            Direction = np.array([x/np.linalg.norm(x) for x in Direction])

        # Simulate photons according to one of two laser calibration schemes:
        # Scheme 2 = LaserCalibrationAnode: Photons are emitted from 4 different fibers 
        # which are located at the edge of the anode plane where due to the geometric 
        # packing of square tiles in a circular shape we have some dead space.
        # Photons in this configuration will be emitted downwards towards the cathode 
        elif self.Yaml['Simulation']['PhotonLocation'] == 'LaserCalibrationAnode':
            # Get inner radius of field rings and shorten by 0.95 to have photons not point directly at them
            InnerRadius = self.Detector.GetMinRadius(np.sum(self.Detector.GeoData['Meshes']['Field-Shaping-Rings'])) * 0.95

            # Get bottom Z-position of the anode to place the fiber, minus 1mm so photons don't start inside anode 
            AnodeBottom = self.Detector.GetMinHeight(np.sum(self.Detector.GeoData['Meshes']['Anode'])) - 1.0

            # Define equally spaces angles around anode and convert polar coordinates to cartesian coordinates
            Phi = np.linspace(0,2*np.pi,5)[:4]+np.pi/4.0
            X = np.mean(InnerRadius) * np.cos(Phi)
            Y = np.mean(InnerRadius) * np.sin(Phi)
            Z = np.array([AnodeBottom]*len(X))
            Points = np.dstack((X,Y,Z))[0]

            # Photons are pointing down towards the cathode
            Direction = np.tile((0,0,-1), (len(X),1))
        elif 'Point=' in self.Yaml['Simulation']['PhotonLocation']:
            Point = self.Yaml['Simulation']['PhotonLocation'].split('=')[1]
            if ',' in Point:
               Point = np.array(Point.split(','), dtype=np.float)#should just be x,y,z coords
            else:
                print('The correct format for simulating at a point is Point= x,y,z')
                sys.exit()
            Points = np.array([Point]) #needs to be 2D array
            print("Source locations: ", Points)
            Direction = [None]*self.Yaml['Simulation']['NumberOfSources']
        elif 'MuonFile' in self.Yaml['Simulation']['PhotonLocation']:
            #add relative path
            MuonFile = self.Yaml['Simulation']['Generator'].split('=')[1][1:-1]
            print("Looking for muon input at:", MuonFile, ", found:", os.path.isfile(MuonFile))
            ###read file and save muon entry positions here###
            # Should return 2D array with columns:
            # [data types, muon #, Angle [deg], energy [GeV], entry [mm], exit [mm]]
            MuonData = ReadMuonsFromCSV(MuonFile)
            #print(MuonData) 

            #to use once we have real muon inputs
            Points = MuonData[4]
            ExitPoints = MuonData[5]
            Angle = MuonData[2]
            Energy = MuonData[3]

            #just data for testing with right now
            #Points = np.array([[0,0,6650.0], [3000,3000,4000], [-3500, 0, 0]])
            #ExitPoints = np.array([[0,0,-6650.0], [3000,3000,-5000], [4500, 0, 0]])
            #Angle = np.array([10, 12, 15])
            #Energy = np.array([1.0, 1.5, 2.5])

            #anything only 1D is going to need padding to be able to go into an array with the 3D ones
            Angle = [[a,0,0] for a in Angle]
            Energy = [[E,0,0] for E in Energy]

            #Shift the positions as the input will be wrt to the WT center and
            #not the chroma weighted center which we need here
            GeoShift = self.Detector.GeoData['Center']
            WT_Height = 13300.0 #total height of the WT
            shift_origin = np.array([GeoShift[0], GeoShift[1], GeoShift[2] + WT_Height/2])
            
            Points = Points + shift_origin
            ExitPoints = ExitPoints + shift_origin
            
            print(Points)

            #load all data into our array to pass to the photon generator
            ExtraData=np.array([Points, ExitPoints, Angle, Energy])
            Direction = [None]*len(Points)

            
        elif 'MuonGen' in self.Yaml['Simulation']['PhotonLocation']:
            numOfMuons = self.Yaml['Simulation']['NumberOfSources']
            file = MuonGen.MuonGen2(numOfMuons, 'Output.csv')
            MuonData = ReadMuonsFromCSV2(MuonFile)
            #print(MuonData) 

            #to use once we have real muon inputs
            Points = MuonData[3]
            ExitPoints = MuonData[4]
            Angle = MuonData[1]
            Energy = MuonData[2]

            #just data for testing with right now
            #Points = np.array([[0,0,6650.0], [3000,3000,4000], [-3500, 0, 0]])
            #ExitPoints = np.array([[0,0,-6650.0], [3000,3000,-5000], [4500, 0, 0]])
            #Angle = np.array([10, 12, 15])
            #Energy = np.array([1.0, 1.5, 2.5])

            #anything only 1D is going to need padding to be able to go into an array with the 3D ones
            Angle = [[a,0,0] for a in Angle]
            Energy = [[E,0,0] for E in Energy]

            #Shift the positions as the input will be wrt to the WT center and
            #not the chroma weighted center which we need here
            GeoShift = self.Detector.GeoData['Center']
            WT_Height = 13300.0 #total height of the WT
            shift_origin = np.array([GeoShift[0], GeoShift[1], GeoShift[2] + WT_Height/2])
            
            Points = Points + shift_origin
            ExitPoints = ExitPoints + shift_origin
            
            print(Points)

            #load all data into our array to pass to the photon generator
            ExtraData=np.array([Points, ExitPoints, Angle, Energy])
            Direction = [None]*len(Points) 
            
        else:
            print("You need to specify a source location type.")
            sys.exit()

        self.GenPos = np.array(Points)
        self.AddAllSources(self.GenPos, Direction, ExtraData)    

    def AddAllSources(self, PhotonSources, Direction=None, ExtraData=None):
        if 'PhotonDirection' in self.Yaml['Simulation']: 
            Direction = tuple(self.Yaml['Simulation']['PhotonDirection'])#['NumberOfPhotons']
        
        PhotonEvents = []
        if isinstance(Direction, np.ndarray):
            for ii,Position in enumerate(PhotonSources):
                Event = Photons.UseGenerator(NPhotons = self.Yaml['Simulation']['NumberOfPhotons'], \
                    Position = Position, \
                    Direction = Direction[ii], \
                    Wavelength = 178.0, \
                    Generator = self.Yaml['Simulation']['Generator'])
                PhotonEvents.append(Event)
        else:
            if 'PhotonWavelength' in self.Yaml['Simulation']:
                Wavelength = self.Yaml['Simulation']['PhotonWavelength']
                print("Wavelength set to: "+ str(Wavelength))
            else:
                Wavelength = 193.0

            NPhotonsArray = []
            if ExtraData is None:
                for ii,Position in enumerate(PhotonSources):
                    Event = Photons.UseGenerator(NPhotons = self.Yaml['Simulation']['NumberOfPhotons'], \
                        Position = Position, \
                        Direction = Direction, \
                        Wavelength = Wavelength, \
                        Generator = self.Yaml['Simulation']['Generator'])
                    PhotonEvents.append(Event)

            else: #here we are passing UseGenerator extra data from the yaml file that is needed
                for ii,Position in enumerate(PhotonSources):
                    Event = Photons.UseGenerator(NPhotons = self.Yaml['Simulation']['NumberOfPhotons'], \
                        Position = Position, \
                        Direction = Direction, \
                        Wavelength = Wavelength, \
                        Generator = self.Yaml['Simulation']['Generator'], ExtraData = ExtraData[:,ii])
                    PhotonEvents.append(Event)
                    
                    #these runs may have different numbers of photons and the meta data saved needs to reflect that
                    NPhotonsArray.append(len(Event.pos))

                self.Yaml['Simulation']['NumberOfPhotons'] = NPhotonsArray

        self.PhotonEvents = PhotonEvents

    def PrintPhotonHistory(self, ev, PhotonFlags):
        for ii,(x,y) in enumerate(zip(self.PhotonTags,PhotonFlags)): 
            Value = len(ev.photons_end.pos[y])
            Value /= float(self.Yaml['Simulation']['NumberOfPhotons'])
            Print(x, Value)
        Print('-')

    def GetPhotonFlags(self, ev):
        PhotonFlags = []
        PhotonFlags.append((ev.flags & (0x1 << 0)).astype(bool))    #NO_HIT
        PhotonFlags.append((ev.flags & (0x1 << 1)).astype(bool))    #BULK_ABSORB
        PhotonFlags.append((ev.flags & (0x1 << 2)).astype(bool))    #SURFACE_DETECT
        PhotonFlags.append((ev.flags & (0x1 << 3)).astype(bool))    #SURFACE_ABSORB
        PhotonFlags.append((ev.flags & (0x1 << 4)).astype(bool))    #RAYLEIGH_SCATTER
        PhotonFlags.append((ev.flags & (0x1 << 5)).astype(bool))    #REFLECT_DIFFUSE
        PhotonFlags.append((ev.flags & (0x1 << 6)).astype(bool))    #REFLECT_SPECULAR
        PhotonFlags.append((ev.flags & (0x1 << 7)).astype(bool))    #SURFACE_REEMIT
        PhotonFlags.append((ev.flags & (0x1 << 8)).astype(bool))    #SURFACE_TRANSMIT
        PhotonFlags.append((ev.flags & (0x1 << 9)).astype(bool))    #BULK_REEMIT
        PhotonFlags.append((ev.flags & (0x1 << 10)).astype(bool))   #MATERIAL_REFL
        PhotonFlags.append((ev.flags & (0x1 << 31)).astype(bool))   #NAN_ABORT
        return PhotonFlags

    def GetSource(self, PhotonSource): 
        event = Photons.PhotonBomb(int(self.Yaml['Simulation']['NumberOfPhotons']), 178, PhotonSource)
        return event
    
    def GetIncidentAngles(self, PhotonDirection, PhotonLastHitTriangles, DetectorPos):
        IncidentAngles = []
        for ii,t in enumerate(self.Triangles[PhotonLastHitTriangles]): 
            v1 = t[1] - t[0]
            v2 = t[2] - t[1]
            SurfaceNormal = np.cross(v1, v2)
            SurfaceNormal /= np.linalg.norm(SurfaceNormal)
            IncidentAngle = np.dot(-PhotonDirection[ii], SurfaceNormal)
            IncidentAngle = np.arccos(IncidentAngle)/np.pi*180
            #print(IncidentAngle)
            IncidentAngles.append(IncidentAngle)
            #Plot.PlotIncidentAngle(DetectorPos, SurfaceNormal, PhotonDirection, t, ii)
        return np.array(IncidentAngles)

    def GetVariablesToSave(self, ev): 
        self.Variables = ['DetectionFlag', 'NumDetected', 'ChannelCharges', 'ChannelTimes', 'ChannelIDs', 'NumHitChannels', 'IncidentAngles', 'Flags', 'LastHitTriangle', 'FinalPosition', 'InitialPosition', 'DetectedPos', 'PhotonWavelength']
        if  self.Yaml['Simulation']['SaveVariables'] == 'All':
            pass 
        else: 
            self.Variables = self.Yaml['Simulation']['SaveVariables']
        H5Data = {}
        DetectionFlag = (ev.photons_end.flags & (0x1 << 2)).astype(bool)
        for Variable in self.Variables: 
            if Variable in H5Data.keys():
                continue 
            if Variable == 'NumDetected': 
                Detected = ev.photons_end[DetectionFlag]
                H5Data[Variable] = len(Detected)
            elif 'Channel' in Variable: 
                ChannelHits, ChannelTimes, H5Data['ChannelCharges'] = ev.channels.hit_channels()
                H5Data['ChannelIDs'] = self.Detector.Detector.channel_index_to_channel_id[ChannelHits[0]]
                H5Data['NumHitChannels'] = np.array([len(H5Data['ChannelIDs'])])
            elif Variable == 'IncidentAngles': 
                H5Data['DetectedPos'] = ev.photons_end.pos[DetectionFlag]
                H5Data['DetectorHit'] = self.FindClosestDetector(H5Data['DetectedPos'])
                H5Data[Variable] = self.GetIncidentAngles(ev.photons_end.dir[DetectionFlag], ev.photons_end.last_hit_triangles[DetectionFlag], H5Data['DetectorHit'])
            elif Variable == 'Flags': 
                H5Data[Variable] = ev.photons_end.flags
            elif Variable == 'InitialPosition':
                H5Data[Variable] = ev.photons_beg.pos
            elif Variable == 'FinalPosition': 
                H5Data[Variable] = ev.photons_end.pos
            elif Variable == 'LastHitTriangle': 
                H5Data[Variable] = ev.photons_end.last_hit_triangles
            elif Variable == 'PhotonWavelength':
                H5Data[Variable] = ev.photons_end.wavelengths
            else: 
                H5Data[Variable] = None
        return H5Data

    def Propagation(self, sim, Photons, run, f5): 
        for jj, ev in enumerate(sim.simulate(Photons, keep_photons_beg=True, keep_photons_end=True, keep_hits=False, run_daq=True, max_steps=1000, photons_per_batch=10000000)):
            if jj == 0: 
                Print('Simulation finished', time.clock())

            H5Data = self.GetVariablesToSave(ev)
            f5.WriteEvent(Num = 'Run'+str(run+1).zfill(2)+'_Event'+str(jj+1).zfill(3), 
                        Origin = self.GenPos[jj], 
                        H5Data = H5Data)

            if self.Detector.Job:        
                ProgressBar(jj, int(self.Yaml['Simulation']['NumberOfSources']), '  Exporting events to file:')

    def PropagationStep(self, sim, Events, run, f5): 
        for ii, ev in enumerate(Events):
            print("Run: ", ii+1, " of ", len(Events))
            
            if isinstance(self.Yaml['Simulation']['NumberOfPhotons'], (np.ndarray, list) ):
                NumPhotons = int(self.Yaml['Simulation']['NumberOfPhotons'][ii])
            else:
                NumPhotons = int(self.Yaml['Simulation']['NumberOfPhotons'])

            AllPhotons, PhotonTrack = self.propagate_photon(ev, NumPhotons, 100, self.Detector.Detector)
            if 'PathPlot' in self.Yaml['Simulation']: 
                if self.Yaml['Simulation']['PathPlot']:
                    Plot.PlotEventDisplay(PhotonTrack, NumPhotons, self.GenPos[ii], self.DetectorPos)
                else: #just save photon position to the text file, don't plot anything
                    
                    fileSave=open('photonPath.txt','a')
                    for i in range(NumPhotons):
                        PhotonStart = ev.pos[i]
                        X, Y, Z = PhotonTrack[:,i,0], PhotonTrack[:,i,1], PhotonTrack[:,i,2]
                        X = np.insert(X, 0, PhotonStart[0], axis=0)
                        Y = np.insert(Y, 0, PhotonStart[1], axis=0)
                        Z = np.insert(Z, 0, PhotonStart[2], axis=0)

                        fileSave.write("\n#######\n")
                        np.savetxt(fileSave, X, newline=", ")
                        fileSave.write("\n")
                        np.savetxt(fileSave, Y, newline=", ")
                        fileSave.write("\n")
                        np.savetxt(fileSave, Z, newline=", ")
                    
                    fileSave.write("\n#******New Event*******#\n")
                    fileSave.write("#    #\n")
                    fileSave.write("#*************#\n")
                    fileSave.close()
            else:
                Plot.PlotEventDisplay(PhotonTrack, NumPhotons, ev.pos, self.DetectorPos)
                plt.savefig('display_%d.png' % ii, bbox_inches='tight', dpi=300)
                plt.close()
            # Plot.PlotTravelDistance(PhotonTrack, int(self.Yaml['Simulation']['NumberOfPhotons']))
            # print(len(PhotonTrack))
            # print([len(ii) for ii in PhotonTrack])
            # H5Group = f5.CreateGroup(ii)
            # for jj, Photons in enumerate(AllPhotons):
            #    PhotonFlags = self.GetPhotonFlags(Photons)
            #    PhotonHits = [Photons.pos[x] for x in PhotonFlags]
            #    PhotonDirectionAll = Photons.dir
            #    PhotonLastHitTrianglesAll = Photons.last_hit_triangles
            #    DetectorPos = self.FindClosestDetector(PhotonHits[2])
            #    IncidentAnglesAll = self.GetIncidentAngles(PhotonDirectionAll, PhotonLastHitTrianglesAll, DetectorPos) 

            #    H5SubGroup = f5.CreateSubGroup(Group=H5Group, Name=jj)
            #    f5.WriteEvent(Num = 'Run'+str(run+1).zfill(2)+'_Event'+str(jj+1).zfill(3), 
            #            Origin = self.GenPos[ii], 
            #            FinalPosition = Photons.pos, 
            #            DetectorHit = DetectorPos,
            #            IncidentAngles = IncidentAnglesAll,
            #            LastHitTriangle = PhotonLastHitTrianglesAll,
            #            Flags = Photons.flags) # ,
            #            New = False,
            #            Group = H5SubGroup)

    def GetName(self): 
        now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        OutputName = ''
        OutputName += self.Yaml['Simulation']['OutputPath']
        OutputName += 'chroma_'
        OutputName += self.Yaml['Detector']['DetectorName']
        OutputName += '_'+self.Yaml['Simulation']['OutputFilename']
        OutputName += '_'+now
        OutputName += '_r'+str(random.randint(1000, 10000))
        OutputName += '.h5'
        return OutputName

    def Simulate(self):        
        OutputName = self.GetName()
        f5 = Output.H5Writer(OutputName)
        f5.SaveOpticalParameters(self.OpticalParameters)
        f5.WriteMetaData(self.Yaml)
        f5.WriteAttribute(Name='TagNames', Attribute=list(self.Detector.GeoData['Triangles'].keys()))
        f5.WriteAttribute(Name='TagValues', Attribute=[self.Detector.GeoData['Triangles'][x] for x in list(self.Detector.GeoData['Triangles'].keys())])
        if 'FiducialVolume' in self.Yaml['Detector'].keys(): 
            f5.WriteAttribute(Name='FVHeight', Attribute=self.Detector.GeoData['FiducialHeight'])
            f5.WriteAttribute(Name='FVRadius', Attribute=self.Detector.GeoData['FiducialRadius'])
        if 'SkinVolume' in self.Yaml['Detector'].keys(): 
            f5.WriteAttribute(Name='SkinHeight', Attribute=self.Detector.GeoData['SkinHeight'])
            f5.WriteAttribute(Name='SkinRadius', Attribute=self.Detector.GeoData['SkinRadius'])
        f5.WriteAttribute(Name='GeometryShift', Attribute=self.Detector.GeoData['Center'])

        Print('=')
        for key in self.Yaml['Simulation'].keys(): 
            #print(key, self.Yaml['Simulation'][key])
            Print(key, self.Yaml['Simulation'][key])
        Print('-')
        
        for run in range(int(self.Yaml['Simulation']['NumberOfRuns'])):
            Print('Run',run)
            self.GeneratePhotons()
            Print('Photon generation done', time.clock())
            sim = chroma.sim.Simulation(self.Detector.Detector, geant4_processes=0, seed=np.random.randint(1,1000))
            Print('Simulation process starting', time.clock())
            if self.Yaml['Simulation']['PropagationMode'] == 'Total':
                self.Propagation(sim, self.PhotonEvents, run, f5)
            if self.Yaml['Simulation']['PropagationMode'] == 'Step':
                self.PropagationStep(sim, self.PhotonEvents, run, f5)
        f5.WriteMetaData(self.Yaml)
        f5.Close()

    def propagate_photon(self, photon_type, numPhotons, nr_steps, geometry):
        nthreads_per_block = 64
        max_blocks = 1024
        rng_states = chroma.gpu.get_rng_states(nthreads_per_block*max_blocks) 
        gpu_photons = chroma.gpu.GPUPhotons(photon_type);
        gpu_geometry = chroma.gpu.GPUGeometry(geometry)
        PhotonTrack = []
        AllPhotons = [] 
        Temp = np.zeros((numPhotons, 3))
        for ii in range(nr_steps):
            start_sim = time.clock()
            gpu_photons.propagate(gpu_geometry, rng_states, nthreads_per_block=nthreads_per_block, max_blocks=max_blocks, max_steps=1)
            photons = gpu_photons.get()
            if np.array_equal(Temp, photons.pos):
                break
            else:
                Temp = photons.pos 
                pass 
            # print(photons.pos)
            AllPhotons.append(photons)
            PhotonTrack.append(photons.pos)
        return AllPhotons, np.array(PhotonTrack)

    def FindClosestDetector(self, PhotonPosition):
        self.DetectorPosTree = scipy.spatial.cKDTree(self.DetectorPos)
        QueryResults = self.DetectorPosTree.query(PhotonPosition)
        ClosestTriangleIndex = QueryResults[1].tolist()
        ClosestTriangleDistance = QueryResults[0].tolist()
        DetectorPos = self.DetectorPos[ClosestTriangleIndex]
        return DetectorPos

def get_perp(x):
    """Returns an arbitrary vector perpendicular to `x`."""
    a = np.zeros(3)
    a[np.argmin(abs(x))] = 1
    return np.cross(a,x)
