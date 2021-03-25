import sys
import os

#folder to longthrow
folder = sys.argv[1]

if folder[-1] != '/':
    folder = folder + '/'

for r, d, f in os.walk(folder):
    for file in f:
        plyname = r + file.split(".")[0] + ".ply"

        if '.obj' in file and file != "out.obj" and not os.path.exists(plyname):
            print("Processing file: " + file)
            filename = r + file
            f = open(filename, 'r')
            data = f.read()
            f.close()
            out = open(plyname, 'w')
            vertices = ""
            numOfVerts = 0
            lines = data.split('\n')
            for line in lines:
                if line.startswith('v '):
                    vertices = vertices + line[2:] + '\n'
                    numOfVerts = numOfVerts + 1

            header = "ply\nformat ascii 1.0\nelement vertex " + str(numOfVerts) + "\nproperty float x\nproperty float y\nproperty float z\nend_header\n"

            out.write(header + vertices)
            out.close()
