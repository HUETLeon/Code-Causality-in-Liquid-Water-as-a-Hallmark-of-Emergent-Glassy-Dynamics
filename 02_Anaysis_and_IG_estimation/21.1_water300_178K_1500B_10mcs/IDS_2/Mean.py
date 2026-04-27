import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', help='name of the file to mean')

args=parser.parse_args()

Name=args.name

data = np.loadtxt(Name)

out = data.mean(axis=0)

outputname = Name.split('.')[0]+'_mean.txt'

Number = (Name.split('E')[1]).split('.')[0]

with open(outputname, 'w') as file:
    file.write(f"{Number}\t{out[0]}\t{out[1]}\t{out[2]}\n")




