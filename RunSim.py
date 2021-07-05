import sys 
sys.path.insert(0, './Simulation')

import time, argparse, yaml
import Multisim
from Utilities import Print, Initialize

parser = argparse.ArgumentParser()
parser.add_argument("-y", type=str, action="store", dest="Yaml")
parser.add_argument("-v", action="store_true", dest="view", default=False)
parser.add_argument("-d", action="store_true", dest="debug", default=False)
parser.add_argument("-j", action="store_false", dest="job", default=True)
parser.add_argument("-s", action="store_true", dest="submit", default=False)
Input = parser.parse_args()

if __name__ == '__main__':
    StartNow = Initialize() 
    Yaml = yaml.safe_load(open(Input.Yaml, 'r'))
    Multi = Multisim.CheckChanges(Yaml)
    if Multi == True: 
        Multisim.CreateYamlCards(Yaml)
    else: 
        import Detector, Simulation
        Det = Detector.Detector(Input)
        Det.ChangeOpticalProperties()
        Det.Build()
        Print('Time elapsed in seconds', time.clock())

        if(Input.view): 
            Det.View()

        Sim = Simulation.Simulation(Det)
        Sim.Simulate()

    Print('=')
    Print('Time elapsed in seconds', time.clock())
