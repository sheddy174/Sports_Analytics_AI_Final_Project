import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):

      self.minimum_distance = 3

      first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      self.first_frame_gray = first_frame_grayscale
      self.lk_params = dict(winSize=(15, 15),
                          maxLevel=2,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

      mask_features = np.zeros_like(first_frame_grayscale)
      self.mask_features = mask_features
      mask_features[:,0:20]=1
      mask_features[:,900:1000]=1

      self.features = dict(
          maxCorners=100,
          qualityLevel=0.3,
          minDistance=7,
          blockSize=7,
          mask=mask_features
      )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
      for object, object_tracks in tracks.items():
        for frame_num, track in enumerate(object_tracks):
          for track_id, track_info in track.copy().items():  # Create a copy of the dictionary
            position = track_info['position']  # Get the original position
            camera_movement = camera_movement_per_frame[frame_num]
            position_adjusted = (position[0] -camera_movement[0], position[1]-camera_movement[1])
            track_info['position_adjusted'] = position_adjusted  # Add the adjusted position to the track_info
            
      return tracks
    # removed player_positions

    
    def getCameraMovement(self, frames,read_from_stub=False, stub_path=None):
      #Read from stub
      if read_from_stub:
        if stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Stub file {stub_path} not found. Computing camera movement from scratch.")

      # Compute camera movement from scratch
      camera_movement = [[0,0]]*len(frames)
      old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
      old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)  ## _ replaced with (.)

      if old_features is None:
        print("No features found in the first frame.")
        return camera_movement

      for frame_num, frame in enumerate(frames[1:], start=1):
        frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
        new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params) ##added self and dot(.)
        
        if new_features is None:
            print(f"No features found in frame {frame_num}.")
            continue

        max_distance = 0
        camera_movement_x, camera_movement_y = 0,0

        for i, (new,old) in enumerate(zip(new_features, old_features)):
          new_feature_point = new.ravel()
          old_feature_point = old.ravel()
          distance = measure_distance(new_feature_point,old_feature_point)
          if distance > max_distance:
            max_distance = distance
            camera_movement_x, camera_movement_y = measure_xy_distance(new_feature_point, old_feature_point) # removed the s
        if max_distance > self.minimum_distance:
          camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
          old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

        old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

      return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
      output_frames=[]
            ## added an "s to frame"
      for frame_num, frame in enumerate(frames):
        frame = frame.copy()

        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        camera_movement_x, camera_movement_y = camera_movement_per_frame[frame_num]
        frame = cv2.putText(frame, f"Camera Movement X:  {camera_movement_x:.2f}", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        frame = cv2.putText(frame, f"Camera Movement Y:  {camera_movement_y:.2f}", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        output_frames.append(frame)

      return output_frames

