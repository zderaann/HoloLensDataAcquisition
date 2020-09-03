import sys
import os
import glob
import tarfile
import argparse
import sqlite3
import shutil
import json
import subprocess
import urllib.request
import numpy as np
from PIL import Image


# Downloading data from HoloLens.
# Arguments: Username for HoloLens Portal,
#            Password for HoloLens Portal,
#            Folder, where to download,
#            HoloLensIP(127.0.0.1:10080, if using USB),
#            bool: Delete after download (1 = delete)
# Example:
#         python HoloLensDataExtraction.py "username" "password" "D:/Documents/HoloLensData/" "127.0.0.1:10080" "1"

recording_folders = []


def connect(address, username, password):
    print("Connecting to HoloLens Device Portal...")
    url = "http://{}".format(address)
    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, url, username, password)
    handler = urllib.request.HTTPBasicAuthHandler(password_manager)
    opener = urllib.request.build_opener(handler)
    opener.open(url)
    urllib.request.install_opener(opener)

    print("=> Connected to HoloLens at address:", url)

    response = urllib.request.urlopen(
        "{}/api/app/packagemanager/packages".format(url))
    packages = json.loads(response.read().decode())

    package_full_name = None
    for package in packages["InstalledPackages"]:
        if package["Name"] == "CV: Recorder":
            package_full_name = package["PackageFullName"]
            break
    assert package_full_name is not None, \
        "App not found"

    print("=> Found application with name:",
          package_full_name)

    print("Searching for recordings...")

    response = urllib.request.urlopen(
        "{}/api/filesystem/apps/files?knownfolderid="
        "LocalAppData&packagefullname={}&path=\\\\TempState".format(
            url, package_full_name))
    recordings = json.loads(response.read().decode())

    recording_names = []
    for recording in recordings["Items"]:
        # Check if the recording contains any file data.
        response = urllib.request.urlopen(
            "{}/api/filesystem/apps/files?knownfolderid="
            "LocalAppData&packagefullname={}&path=\\\\TempState\\{}".format(
                url, package_full_name, recording["Id"]))
        files = json.loads(response.read().decode())
        if len(files["Items"]) > 0:
            recording_names.append(recording["Id"])
    recording_names.sort()

    print("=> Found a total of {} recordings".format(
        len(recording_names)))
    return url, recording_names, package_full_name


def download_recordings(url, package_full_name, recordings, workspace_path):
    for recording_name in recordings:
        if recording_name is None:
            return

        recording_folders.append(recording_name)
        recording_path = os.path.join(workspace_path, recording_name)
        mkdir_if_not_exists(recording_path)

        print("Downloading recording {}...".format(recording_name))

        response = urllib.request.urlopen(
            "{}/api/filesystem/apps/files?knownfolderid="
            "LocalAppData&packagefullname={}&path=\\\\TempState\\{}".format(
                url, package_full_name, recording_name))
        files = json.loads(response.read().decode())

        for file in files["Items"]:
            if file["Type"] != 32:
                continue

            destination_path = os.path.join(recording_path, file["Id"])
            if os.path.exists(destination_path):
                print("=> Skipping, already downloaded:", file["Id"])
                continue

            print("=> Downloading:", file["Id"])
            urllib.request.urlretrieve(
                "{}/api/filesystem/apps/file?knownfolderid=LocalAppData&" \
                "packagefullname={}&filename=\\\\TempState\\{}\\{}".format(
                    url, package_full_name,
                    recording_name, file["Id"]), destination_path)


def delete_recordings(url, package_full_name, recordings):
    for recording_name in recordings:
        if recording_name is None:
            return

        print("Deleting recording {}...".format(recording_name))

        response = urllib.request.urlopen(
            "{}/api/filesystem/apps/files?knownfolderid="
            "LocalAppData&packagefullname={}&path=\\\\TempState\\{}".format(
                url, package_full_name, recording_name))
        files = json.loads(response.read().decode())

        for file in files["Items"]:
            if file["Type"] != 32:
                continue

            print("=> Deleting:", file["Id"])
            urllib.request.urlopen(urllib.request.Request(
                "{}/api/filesystem/apps/file?knownfolderid=LocalAppData&" \
                "packagefullname={}&filename=\\\\TempState\\{}\\{}".format(
                    url, package_full_name,
                    recording_name, file["Id"]), method="DELETE"))



def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory " + path + " created")

#Arguments: Username for HoloLens Portal, Password for HoloLens portal, Folder, where to download, HoloLensIP, Delete after dowload

def convert_images(folder):
    for r, d, f in os.walk(folder):
        for file in f:
            # print(file)
            if '.pgm' in file or '.ppm' in file:
                print(file)
                if '.pgm' in file:
                    img = Image.open(r + "/" + file)
                    img = img.transpose(Image.ROTATE_270)
                else:
                    f = open(r + "/" + file, 'rb')
                    line = f.readline()
                    line = f.readline()
                    width = int(line.split(b' ')[0])
                    height = int(line.split(b' ')[1])
                    line = f.readline()
                    data = f.read()

                    img = Image.frombytes("RGB", (width, height), data, "raw", "BGRX", 0, 1)
                    f.close()
                filename = r + "/" + file.split(".")[0] + ".jpg"
                print("saving  as: " + filename)
                img.save(filename)



username = sys.argv[1]
password = sys.argv[2]
folder = sys.argv[3]
if not folder[-1] == "/":
    folder = folder + "/"
IP = sys.argv[4]
delAfter = sys.argv[5]

mkdir_if_not_exists(folder)

url, records, package_full_name = connect(IP, username, password)
download_recordings(url, package_full_name, records, folder)

if delAfter == "1":
    delete_recordings(url, package_full_name, records)

#extract tars
sensors = ["pv", "vlc_ll", "vlc_lf", "vlc_rf", "vlc_rr", "long_throw_depth"]

print("Extracting images")
for recording in recording_folders:
    for sensor in sensors:
        tarpath = folder + recording + "/" + sensor + ".tar"
        if os.path.isfile(tarpath):
            print("Extracting from: " + tarpath)
            tar = tarfile.open(tarpath)
            tar.extractall(folder + recording + "/")

    # convert PV and rotate vlc images
    print("Converting image files")
    cams = sensors[0:5]
    for cam in cams:
        print("Converting files in: " + folder + recording + "/" + cam)
        convert_images(folder + recording + "/" + cam)



