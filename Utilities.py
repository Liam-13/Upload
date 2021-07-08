import sys, os, datetime, numpy, matplotlib, time, csv
from decimal import Decimal
import numpy as np
    
try:
    rows, columns = os.popen('stty size', 'r').read().split()
except:
    rows, columns = 80, 100 

def ProgressBar(count, total, phrase):
    barLength = 15 # Modify this to change the length of the progress bar
    status = '%d/%d' % (count+1, total)
    progress = (count+1)/float(total)
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if count == total-1:
        progress = 1
        status = status+"           \n"
    block = int(round(barLength*progress))
    text = '\r%-*s [%s] %-*s %s' % (47,phrase,"#"*block+"-"*(barLength-block),8,'%.2f%%'%Decimal(progress*100),status)
    sys.stdout.write(text)
    sys.stdout.flush()

def Print(String,Value='',Verbose=-1):
    if Verbose > 0:
        length = 2+45+len(str(Value))
        offset = 2 + 2
        width = int(columns) - len(str(Value)) - offset
        OutString = ''
        if isinstance(Value, str):
            OutString = '  %-*s %s' % (width, String, Value)
        elif isinstance(Value, int):
            OutString = '  %-*s %s' % (width, String, Value)
        elif isinstance(Value, float):
            width = int(columns) - len("{:.3f}".format(Value)) - offset
            OutString = '  %-*s %s' % (width, String, "{:.3f}".format(Value))
        elif isinstance(Value, list):
            width = int(columns) - len('[%s]'% ','.join(Value)) - offset
            OutString = '  %-*s %s' % (width, String, '[%s]'% ','.join(Value))
        elif hasattr(Value, "__len__"):
            width = int(columns) - len(str(np.round(Value,2)).strip(' ')) - offset
            OutString = '  %-*s %s' % (width, String, str(np.round(Value,2)).strip(' '))
        if len(String)==1:
            OutString = String*int(columns)
        print(OutString)
    else:
        return 0

def Initialize(): 
    now = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')
    Print('=')
    Print('Matplotlib version', matplotlib.__version__)
    Print('Numpy version', numpy.__version__)
    Print('=')
    Print('Date and Time',now)
    return time.time()

def CreateDir(Path):
    if not os.path.exists(Path):
        try:
            os.makedirs(Path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

# https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
def Sort(Files): 
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(Files, key = alphanum_key)

def ReadMuonsFromCSV(csvFile):
    #this will hold 6 arrays, one for each type of data, but using a list over all
    #to hold them so we can have varying lengths
    data=[]

    temp=np.empty(shape=[0,5])
    MuonNum = np.array([])
    Angle = np.array([])
    Energy = np.array([])
    #these two we want to be 2D arrays with the second dimension 
    #being size 3 for our (x,y,z) coords
    Entry = np.empty((0, 3))
    Exit = np.empty((0, 3))

    with open(csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0 #sets the line to 0 so names are names
    
        for row in csv_reader:                                  
            #This finds is a loop looking at each row in the csv_reader
      
            if line_count == 0:                                 
                #print(f'Column names are {", ".join(row)}')     
                #This line prints off the column names so you know what the order is 
                data.append(np.array([r for r in row]))
                #This appends the four column names together in the first array of data
                line_count += 1                                 
                #This is so it can tell how many are being read 
            else:
                count = 0
                for x in row:
                    if len(x) > 10:
                        #Eleminate the square brackets cause they were causing problems
                        x = x.replace("[","")
                        x = x.replace("]","")
          
                        z = x.split(' ')
                        #some times arrays have an extra space added at start for alignment that we need to remove
                        z = [k for k in z if k is not '']
                        z = [float(k) for k in z]
                        #This floats the numbers in a way that works both positions and single numbers
                        count += 1
                    else:
                        z = float(x)
                        count+=1
                    
                    if count == 1:
                        MuonNum = np.append(MuonNum, z)
                        #append the muon number to the data array
                    elif count == 2:
                        Angle = np.append(Angle, z)
                        #append the angle to the data array
                    elif count == 3:
                        Energy = np.append(Energy, z)
                        #append the Energy to the data array
                    elif count == 4:
                        Entry = np.append(Entry, [z], axis=0)
                        #append the Entry to the data array
                    elif count == 5:
                        Exit = np.append(Exit, [z], axis=0)
                        #append the Exit
               
                    line_count += 1

        #we want data to be a 2D list here
        data.append(MuonNum)
        data.append(Angle)
        data.append(Energy)
        #convert these two from m to mm
        data.append(Entry*1000)
        data.append(Exit*1000)
      
    return data
