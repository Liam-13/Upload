
def ReadMuonsFromCSV2(text):

  import numpy as np
  import csv

  #########################
  ####  CSV Reader  #######
  #########################

  '''data=np.array([]) 
  temp=np.empty(shape=[0,5])
  boop = np.array([])
  MuonNum = np.array([])
  Angle = np.array([])
  Energy = np.array([])
  Entry = np.array([])
  Exit = np.array([])'''
  data = [] 
  temp = []
  boop = []
  MuonNum = []
  Angle = []
  Energy = []
  Entry = []
  Exit = []
  with open(text) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0                                          
    #sets the line to 0 so names are names
    
    for row in csv_reader:                                  
    #This finds is a loop looking at each row in the csv_reader
      tempEntry = []
      tempExit = []      
      if line_count == 0:                                 
        #print(f'Column names are {", ".join(row)}')     
        #This line prints off the column names so you know what the order is 
        #for x in range(5):
          #data.append(row[x])
        #This appends the four column names together in the first array of data
        line_count += 1                                 
        #This is so it can tell how many are being read 
      else:
        #print(f'\t Muon # is {row[0]}, Angle [deg] is {row[1]}, energy [GeV] is {row[2]}, entry [mm] is {row[3]}, exit [mm] is {row[4]}.')
        #Prints the information, used to double check that it worked well, will comment out
        '''flt = float(row)
        data.append([row])'''
        count = 0
        for x in row:
          count += 1
          if len(x) > 10:
            x = x.replace("[","")
            #Eleminate the square brackets cause they were causing problemos
         
            x = x.replace("]","")
            
            #Same
          
            z = [float(k) for k in x.split(',')]
            #This floats the numbers in a way that works both positions and single numbers
            #print(z)
            
            #print(z)
          else:
            x = x.replace("[","")
            x = x.replace("]","")
            z = float(x)
            #print(z)
          if count == 1:
            MuonNum.append(z)
            #append the muon number to the data array
          if count == 2:
            Angle.append(z)
            #append the angle to the data array
          if count == 3:
            Energy.append(z)
            #append the Energy to the data array
          if count == 4 or count == 5 or count == 6:
            tempEntry.append(z)
            if count == 6:
              Entry.append(tempEntry)
            #append the Entry to the data array
          if count == 7 or count == 8 or count == 9:
            tempExit.append(z)
            if count == 9:
              Exit.append(tempExit)
            #append the Exit
            
            
        #print(boop)
          line_count += 1
  data.append(MuonNum)
  data.append(Angle)
  data.append(Energy)
  data.append(Entry)
  data.append(Exit)
  return data
      #print(f'Processed {line_count} lines.')
      #prints how many lines are processed, will be commented out
    
