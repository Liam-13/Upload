import tables
import os 
import errno
import warnings
warnings.simplefilter('ignore', tables.NaturalNameWarning)

class H5Writer(object): 
    def __init__(self,Filename):
        self.Filename = Filename
        self.CreateDir()
        self.Filters = tables.Filters(complib='zlib', complevel=5)
        self.File = tables.open_file(self.Filename, 'w', filters=self.Filters)
        self.Label = Label = ['Origin', 'InitialPosition', 'FinalPosition', 'DetectorHit', 'IncidentAngles', 'LastHitTriangle', 'Flags', 'NumDetected', 'ChannelCharges', 'NumHitChannels', 'ChannelIDs', 'DetectedPos', 'PhotonWavelength']
        self.Groups = self.CreateAllGroups()
        self.Arrays = {}

    def WriteEvent(self, Num, Origin, H5Data):
        if 'Origin' not in self.Arrays: 
            self.Arrays['Origin'] = self.File.create_earray(where=self.Groups['Origin'], name='Origin', obj=[Origin])
        else: 
            self.Arrays['Origin'].append([Origin])

        for ii, key in enumerate(H5Data.keys()): 
            x = H5Data[key]
            if x is not None: 
                if key == 'NumDetected': 
                    if key not in self.Arrays: 
                        self.Arrays[key] = self.File.create_earray(where=self.Groups[key], name=key, obj=[x])
                    else:
                        self.Arrays[key].append([x])
                else:
                    atom = tables.Atom.from_dtype(x.dtype)
                    if key not in self.Arrays: 
                        self.Arrays[key] = self.File.create_earray(where=self.Groups[key], name=key, obj=x)
                    else:
                        self.Arrays[key].append(x)
 
    def CreateAllGroups(self): 
        Groups = {}
        for x in self.Label: 
            Groups[x] = self.File.create_group(where='/', name=x)
        return Groups

    def WriteMetaData(self, MetaData): 
        for key in MetaData.keys(): 
            if key in ['Simulation', 'Detector', 'Components']:
                for subkey in MetaData[key].keys():
                    if isinstance(MetaData[key][subkey],dict): 
                        for k,v in MetaData[key][subkey].items(): 
                            if key == 'Components':
                                self.File.set_node_attr(where=self.OPGroup, attrname=subkey+'_'+k, attrvalue=v)
                            else:
                                self.File.set_node_attr(where='/', attrname=subkey+'_'+k, attrvalue=v)
                    else: 
                        self.File.set_node_attr(where='/', attrname=subkey, attrvalue=MetaData[key][subkey])
    
    def SaveOpticalParameters(self, OpticalParameters): 
        self.OPGroup = self.File.create_group('/', 'OpticalParameters')
        for k, v in OpticalParameters.items(): 
            subGroup = self.File.create_group('/OpticalParameters', k)
            '''
            for kk, vv in OpticalParameters[k].items():
                self.File.create_array(where=subGroup, name=kk, obj=vv)
            '''

    def WriteAttribute(self, Name, Attribute):
        self.File.set_node_attr(where='/', attrname=Name, attrvalue=Attribute)
    
    def CreateDir(self):
        Path, File = os.path.split(self.Filename)
        if not os.path.exists(Path):
            try:
                os.makedirs(Path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def Close(self):
        print("Output file: "+self.Filename)
        self.File.close()
