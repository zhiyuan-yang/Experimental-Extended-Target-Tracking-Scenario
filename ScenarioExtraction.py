from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
import numpy as np
import cv2
import os.path as osp
import math

RadarPointCloud.default_filters
nusc = NuScenes(version='v1.0-trainval', dataroot='I:\data/trainval', verbose=True)
s1 = nusc.scene[13]
s2 = nusc.scene[14]
s3 = nusc.scene[15]

sampletoken = s2['first_sample_token']
sample =  nusc.get('sample',sampletoken)
#target no.1 in scene-14
#target no.3 in scene-15
#target no.1 in scene-33
busAnnotationToken = sample['anns'][2]     
timeIndex = 1

groundTruth = np.empty((1,10))
meas = np.empty((1,4))
sensor = np.empty((1,4))

while busAnnotationToken != '':  
    busAnnotation = nusc.get('sample_annotation', busAnnotationToken)
    sample = nusc.get('sample', busAnnotation['sample_token'])
    ETPositon = busAnnotation['translation']    #extended target position in global coordinate
    ETQuar = busAnnotation['rotation']
    global_from_anno = transform_matrix(ETPositon,Quaternion(ETQuar),inverse=True)
    w =  busAnnotation['size'][0]
    l =  busAnnotation['size'][1]
    currGroundTruth = np.hstack((np.array([timeIndex]), ETPositon, ETQuar, np.array([w,l])))
    groundTruth = np.vstack((groundTruth, currGroundTruth))

    
    frontRadarData = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
    ego_pose = nusc.get('ego_pose', frontRadarData['ego_pose_token'])   
    sensor_pose = nusc.get('calibrated_sensor',frontRadarData['calibrated_sensor_token']) #ego-car coordinate
    car_from_sensor = transform_matrix(sensor_pose['translation'], Quaternion(sensor_pose['rotation']), inverse=False)
    global_from_car = transform_matrix(ego_pose['translation'],Quaternion(ego_pose['rotation']), inverse=False)
    trans_matrix = np.dot(global_from_car,car_from_sensor)

    sensor_pose_aug = np.hstack((sensor_pose['translation'],np.array([1]))).reshape(4,1)
    sensorPositionG = np.dot(global_from_car, sensor_pose_aug)   #transform sensor position to global coordinate
    sensorPositionG = sensorPositionG[0:3]
    currSensorPosition = np.hstack((np.array([timeIndex]).reshape(1,1), sensorPositionG.T))
    sensor = np.vstack((sensor, currSensorPosition))
    
    filename = osp.join(nusc.dataroot, frontRadarData['filename'])
    [radarPoints,ndarray] = RadarPointCloud.from_file_multisweep(nusc,sample,'RADAR_FRONT','RADAR_FRONT',5)
                                        #only "cross-moving, unambiguious doppler, valid, artifect less than 25%" is reserved
    radarPoints.transform(trans_matrix)   #transform radar points from sensor-coordinate to global coordinate
    radarPoints = radarPoints.points
    radarPoints = np.vstack((radarPoints[:3,:], np.ones((1, radarPoints.shape[1]))))        
    radarPointsA = np.dot(global_from_anno, radarPoints)                      
    for i in range(0,radarPointsA.shape[1]):
        if -l*0.6 < radarPointsA[0,i] < l*0.6 and  -w*0.6 < radarPointsA[1,i] < w*0.6:      
            currmeas = radarPoints[:3,i]
            currmeas = np.hstack((np.array([timeIndex]),currmeas))  
            meas = np.vstack((meas,currmeas))
    timeIndex = timeIndex + 1
    busAnnotationToken = busAnnotation['next']
np.savetxt('Meas_s2.txt', meas, fmt='%.2f')
np.savetxt('GroundTruth_s2.txt', groundTruth, fmt='%.2f')
np.savetxt('Sensor_s2.txt', sensor, fmt='%.2f')