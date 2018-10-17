# KinectFingertip
legacy kinect fingertip detection

algorithm (implemented with openCV) works as follows:

1. depth thresholding
2. contour extraction
3. approximate contours
4. assume vertices of convex hull to be fingertips if their interior angle is small enough

# update 28.07.2011:

I actually have a new version running with openNI / NITE (on linux) using sekelton tracking as well (currently uses OpenNI-Bin-Linux64-v1.1.0.41).

note: to avoid initial skeleton calibration, it loads a calibration from a file called "UserCalibration.bin". i did not include it into the zip file as its not compatible between different versions of NITE. you can create your own calibration file by calibrating with the openNI user tracker example and press Shift+S (this will generate a UserCalibration.bin, just copy it to the working directory of your binary) 

after that your skeleton will be tracked instantly as you step in front of the kinect. this also works quite well with different people that actually were not calibrated!