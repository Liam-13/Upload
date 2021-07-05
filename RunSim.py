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
    print('started, 1/8')
    Yaml = yaml.safe_load(open(Input.Yaml, 'r'))
    print('yaml Loaded 2/8')
    Multi = Multisim.CheckChanges(Yaml)
    if Multi == True: 
        Multisim.CreateYamlCards(Yaml)
    else: 
        import Detector, Simulation
        print('detector and sim imported 3/8')
        Det = Detector.Detector(Input)
        print('det function 1 done 4/8')
        Det.ChangeOpticalProperties()
        print('det function 2 done 5/8')
        Det.Build()
        print('det function 3 done 6/8')
        Print('Time elapsed in seconds', time.clock())

        if(Input.view): 
            Det.View()

        Sim = Simulation.Simulation(Det)
        print('simulating 1 7/8')
        Sim.Simulate()
        print('simulating 2 8/8')

    Print('=')
    Print('Time elapsed in seconds', time.clock())
