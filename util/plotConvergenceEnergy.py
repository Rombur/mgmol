# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory.
# LLNL-CODE-743438
# All rights reserved.
# This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
# Please also read this link https://github.com/llnl/mgmol/LICENSE
#
#usage: python plotConvergenceEnergy.py mgmol_output
import sys, string
import matplotlib.pyplot as plt

energies=[]

inputfile=open(sys.argv[1],'r')
lines=inputfile.readlines()

flag=0
nst=0
conv_energy=10000.
for line in lines:
  if line.count( 'Number of states'):
    words=line.split()
    nst=eval(words[4])
  num_matches1 = line.count('ENERGY')
  num_matches2 = line.count('%%')
  if num_matches1 & num_matches2:
    words=line.split()
    energy=eval(words[5][:-1])
    energies.append(energy)
    conv_energy=energy

deltaes=[]
for energy in energies:
  deltaes.append((energy-conv_energy)/nst)

plt.plot(deltaes,'r.--')
plt.ylabel('error Eks/orbital [Ry]')
plt.xlabel('outer iterations')
plt.axis([0.,len(deltaes),10.*deltaes[-2],deltaes[0]])
plt.yscale('log')

#plt.show()
plt.savefig('errorEnergy.png', dpi=100)
