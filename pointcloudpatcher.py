import sys
import os
import numpy as np

"""
CSV DATA:
Timestamp,ImageFileName,
Position.X,Position.Y,Position.Z,
Orientation.W,Orientation.X,Orientation.Y,Orientation.Z,
FrameToOrigin.m11,FrameToOrigin.m12,FrameToOrigin.m13,FrameToOrigin.m14,
FrameToOrigin.m21,FrameToOrigin.m22,FrameToOrigin.m23,FrameToOrigin.m24,
FrameToOrigin.m31,FrameToOrigin.m32,FrameToOrigin.m33,FrameToOrigin.m34,
FrameToOrigin.m41,FrameToOrigin.m42,FrameToOrigin.m43,FrameToOrigin.m44,
CameraViewTransform.m11,CameraViewTransform.m12,CameraViewTransform.m13,CameraViewTransform.m14,
CameraViewTransform.m21,CameraViewTransform.m22,CameraViewTransform.m23,CameraViewTransform.m24,
CameraViewTransform.m31,CameraViewTransform.m32,CameraViewTransform.m33,CameraViewTransform.m34,
CameraViewTransform.m41,CameraViewTransform.m42,CameraViewTransform.m43,CameraViewTransform.m44,
CameraProjectionTransform.m11,CameraProjectionTransform.m12,CameraProjectionTransform.m13,CameraProjectionTransform.m14,
CameraProjectionTransform.m21,CameraProjectionTransform.m22,CameraProjectionTransform.m23,CameraProjectionTransform.m24,
CameraProjectionTransform.m31,CameraProjectionTransform.m32,CameraProjectionTransform.m33,CameraProjectionTransform.m34,
CameraProjectionTransform.m41,CameraProjectionTransform.m42,CameraProjectionTransform.m43,CameraProjectionTransform.m44
"""

def quaternion_to_matrix(q):
    R = np.zeros((3,3))
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    R[0][0] = 1 - 2 * qy * qy - 2 * qz * qz
    R[0][1] = 2 * qx * qy - 2 * qz * qw
    R[0][2] = 2 * qx * qz + 2 * qy * qw

    R[1][0] = 2 * qx * qy + 2 * qz * qw
    R[1][1] = 1 - 2 * qx * qx - 2 * qz * qz
    R[1][2] = 2 * qy * qz - 2 * qx * qw

    R[2][0] = 2 * qx * qz - 2 * qy * qw
    R[2][1] = 2 * qy * qz + 2 * qx * qw
    R[2][2] = 1 - 2 * qx * qx - 2 * qy * qy

    return R


folder = sys.argv[1]
csvfilepath = sys.argv[2]

if not folder[-1] == '/':
    folder = folder + '/'

csvfile = open(csvfilepath, 'r')
csvheader = csvfile.readline()
print(csvheader)
csvdata = csvfile.read()
csvfile.close()

parsedcsvdata = csvdata.split("\n")
parsedcsvdata = [x for x in parsedcsvdata if x]
camerainfo = {}
for csvline in parsedcsvdata:
    splitted = csvline.split(",")
    name = splitted[1].split("\\")[-1].split(".")[0]
    camerainfo[name] = csvline


outfile = open(folder + 'out.obj', 'w')
objcount = 1


for r, d, f in os.walk(folder):
    for file in f:
        if '.obj' in file and file != "out.obj":
            print("Processing file: " + file)
            name = file.split(".")[0]
            objfile = open(r + file, 'r')
            objfile.readline()
            objdata = objfile.read()
            objlines = objdata.split("\n")
            objlines = [x.split("v ")[1] for x in objlines if x]
            objfile.close()

            outfile.write('o Object.' + str(objcount) + "\n")
            objcount = objcount + 1

            poseinfo = camerainfo[name]
            poseinfo = poseinfo.split(",")
            position = np.array([float(poseinfo[2]), float(poseinfo[3]), float(poseinfo[4])])
            quaternion = np.array([float(poseinfo[5]), float(poseinfo[6]), float(poseinfo[7]), float(poseinfo[8])])
            rotation = quaternion_to_matrix(quaternion)

            frametoorigin = np.array([[float(poseinfo[9]), float(poseinfo[10]), float(poseinfo[11]), float(poseinfo[12])],
                                      [float(poseinfo[13]), float(poseinfo[14]), float(poseinfo[15]), float(poseinfo[16])],
                                      [float(poseinfo[17]), float(poseinfo[18]), float(poseinfo[19]), float(poseinfo[20])],
                                      [float(poseinfo[21]), float(poseinfo[22]), float(poseinfo[23]), float(poseinfo[24])]])

            cameraviewtransform = np.array([[float(poseinfo[25]), float(poseinfo[26]), float(poseinfo[27]), float(poseinfo[28])],
                                      [float(poseinfo[29]), float(poseinfo[30]), float(poseinfo[31]), float(poseinfo[32])],
                                      [float(poseinfo[33]), float(poseinfo[34]), float(poseinfo[35]), float(poseinfo[36])],
                                      [float(poseinfo[37]), float(poseinfo[38]), float(poseinfo[39]), float(poseinfo[40])]])

            cameraprojectiontransform = np.array([[float(poseinfo[41]), float(poseinfo[42]), float(poseinfo[43]), float(poseinfo[44])],
                                      [float(poseinfo[45]), float(poseinfo[46]), float(poseinfo[47]), float(poseinfo[48])],
                                      [float(poseinfo[49]), float(poseinfo[50]), float(poseinfo[51]), float(poseinfo[52])],
                                      [float(poseinfo[53]), float(poseinfo[54]), float(poseinfo[55]), float(poseinfo[56])]])


            for line in objlines:
                parsed = line.split(" ")
                coords = np.array([float(parsed[0]), float(parsed[1]), float(parsed[2]), 0])
                #coords = np.array([float(parsed[0]), float(parsed[1]), float(parsed[2])])
                #newcoords = np.matmul(rotation, np.transpose(coords)) + np.transpose(position)
                newcoords = np.matmul(np.matmul(frametoorigin, cameraviewtransform), np.transpose(coords))

                outfile.write("v " + str(newcoords[0]) + " " + str(newcoords[1]) + " " + str(newcoords[2]) + "\n")

            outfile.write("\n")



outfile.close()