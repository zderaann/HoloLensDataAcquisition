import sys
import numpy as np
import math
import os

np.set_printoptions(threshold=sys.maxsize)
folder = 'd:/tmp/ARTwinRecording/HoloLensRecording__2021_08_02__11_01_11_MUCLab_2/long_throw_depth'   #sys.argv[1]

if not folder[-1] == '/':
    folder = folder + '/'

uvdatafile = "d:/tmp/ex_models/benchmark/uvdata.txt"   #sys.argv[2]

uvfile = open(uvdatafile, 'r')
uvdata = uvfile.read()
uvlines = uvdata.split("\n")
uvdata = {}

for line in uvlines:
    if not "inf, inf" in line:
        parsed = line.split(" ")
        uvdata[parsed[1] + parsed[2]] = (float(parsed[4].split(",")[0]), float(parsed[5]))


for r, d, f in os.walk(folder):
    for filename in f:
         if ".pgm" in filename:
            print(folder + filename)
            file = open(folder + filename, 'rb')
            timestamp = filename.split(".")[0]

            line = file.readline()
            maxvalue = 65535
            width = 448
            height = 450

            while(not b'65535' in line):
                line = file.readline()

            data = file.read()

            values = np.frombuffer(data, np.short)
            values = np.reshape(values, (height, -1))

            #zpracovat na ply/obj
            # objfile = open(folder + timestamp + ".obj", 'w')
            # objfile.write("o Object.1\n")

            # for i in range(0, 450):
            #     for j in range(0, 448):
            #         r = values[i, j]
            #         if not r == 0:
            #             uv = uvdata["(" + str(j) + "," + str(i) + ")"]
            #             u = uv[0]
            #             v = uv[1]
            #             d = r / math.sqrt(u * u + v * v + 1)
            #             #print(d)
            #             x = d * u
            #             y = d * v
            #             z = d * 1
            #             objfile.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")
            #             #print("Written to file")
            # objfile.close()

            out = open(folder + timestamp + ".ply", 'w')
            vertices = []
            numOfVerts = 0
            for i in range(0, 450):
                for j in range(0, 448):
                    r = values[i, j]
                    if not r == 0 and "(" + str(j) + "," + str(i) + ")" in uvdata:
                        uv = uvdata["(" + str(j) + "," + str(i) + ")"]
                        u = uv[0]
                        v = uv[1]
                        d = r / math.sqrt(u * u + v * v + 1)
                        #print(d)
                        x = d * u
                        y = d * v
                        z = d * 1    
                        vertices.append(str(x) + " " + str(y) + " " + str(z))
                        numOfVerts = numOfVerts + 1
            header = "ply\nformat ascii 1.0\nelement vertex " + str(numOfVerts) + "\nproperty float x\nproperty float y\nproperty float z\nend_header\n"

            out.write(header + "\n".join(vertices))
            out.close()
