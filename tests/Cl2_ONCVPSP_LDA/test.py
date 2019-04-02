#!/usr/bin/env python
import sys
import os
import subprocess
import string

print("Test Cl2_ONCVPSP_LDA...")

mpicmd = sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]
exe = sys.argv[4]
inp = sys.argv[5]
coords = sys.argv[6]
print("coordinates file: %s"%coords)

#create links to potentials files
dst1 = 'pseudo.Cl_ONCVPSP_LDA'
src1 = sys.argv[7] + '/' + dst1

if not os.path.exists(dst1):
  print("Create link to %s"%dst1)
  os.symlink(src1, dst1)

#run mgmol
command = "{} {} -c {} -i {}".format(mpicmd,exe,inp,coords)
output = subprocess.check_output(command,shell=True)

#analyse mgmol standard output
#make sure forces are below tolerance
lines=output.split(b'\n')

tol = 2.e-6
Fz  = 1.2e-3
for line in lines:
  num_matches = line.count(b'%%')
  if num_matches:
    print(line)
  #find output lines with forces
  num_matches = line.count(b'##')
  if num_matches:
    words=line.split()
    if len(words)==8:
      print(line)
      #check forces in x, y directions below tolerance
      for i in range(5,7):
        force = eval(words[i])
        if abs(force)>tol:
          print("force = {}".format(force))
          sys.exit(1)
      #check value of force in z direction
      if abs(eval(words[7])-Fz)>2.e-5:
          print("force in z dir = {}".format(eval(words[7])))
          sys.exit(1)
      else:
        #switch sign for next test
        Fz = -Fz

