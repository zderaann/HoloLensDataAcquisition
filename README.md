HoloLensForCV:Recorder is an App based on HoloLensForCV API released by Microsoft.
It is used to obtain synchronised images from HoloLens sensors, device positions and depth data and save it to the device.
The HoloLensDataExtraction script is used to download all this data from HoloLens.


Open HoloLensForCV with Visual studio 2019. Make sure you have Research mode enabled on your HoloLens.
Set:
	solution configuration to: Release, 
	solution platforms to: x86,
	startup projects to: Recorder(Universal Windows).

In solution explorer, right click Recorder(Universal Windows), click Properties.
Click to Debugging, set Debugger to launch to Remote Machine and type the HoloLens IP address in Machine Name. Click Apply and OK.

Deploy the App by clicking the Remote Machine button with green triangle.

Once the App has started, begin the capture by Air-tapping. The output console in Visual studio will let you know when/if images are being saved.

Air-tap once again to stop the recording. Wait at least a minute to turn off the App (to make sure all the data is saved).  

Connect HoloLens via a USB cable to a computer and run the HoloLensDataExtraction.py. The script requires packages: numpy and PIL.
The script was tested in python version 3.6.5.

Arguments are: Username for HoloLens Portal,
               Password for HoloLens Portal,
               Folder, where to download,
               HoloLensIP(127.0.0.1:10080, if using USB),
               bool: Delete after download (1 = delete)
 Example:
         python HoloLensDataExtraction.py "username" "password" "D:/Documents/HoloLensData/" "127.0.0.1:10080" "1"
         
 If the downloading does not work make sure, you have the newest Windows SDK downloaded and that a process called Windows IP over USB is running.
