import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', help='name of the file to mean')

args=parser.parse_args()

Name=args.name

data = np.loadtxt(Name)

out = data.mean(axis=0)

outputname = Name.split('.')[0]+'_mean.txt'

with open("outputname", 'w') as file:
    file.write(f"{outputname[0]}\t{outputname[1]}\t{outputname[2]}\n")




