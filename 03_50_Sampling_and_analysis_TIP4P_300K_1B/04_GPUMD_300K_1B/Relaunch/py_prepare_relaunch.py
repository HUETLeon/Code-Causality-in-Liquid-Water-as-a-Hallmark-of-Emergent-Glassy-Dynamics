import numpy as np

Last_neig = np.loadtxt("restarts_first_lines.dat") 

WtR = []
SoR = []
for i , l in enumerate(Last_neig):
    print(f"for run {i+1}:")
    Rframe = (l // 5000000 ) * 500000
    print(f"Last Restart frame = {Rframe*10}")
    WtR.append(2500000 + Rframe) #where to restart in the xyzfile
    SoR.append(25000000 - Rframe*10)  #number of state to make
    print(f"The Calculation can be relaunch from the geometrie {WtR[i]} of dump.xyz  for {SoR[i]} fs|steps")

np.savetxt("Where_to_restart.dat", WtR)
np.savetxt("Num_Step_of_restart.dat", SoR)
