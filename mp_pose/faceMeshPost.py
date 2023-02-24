import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

import logging
import rerun as rr
import numpy as np
import numpy.typing as npt
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Final, Iterator, Optional
from rerun.log.annotation import AnnotationInfo, ClassDescription

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    with mp_holistic.Holistic(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
      refine_face_landmarks=True) as holistic:

      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        results = face_mesh.process(image)
        holisitcResults = holistic.process(image)

        # Draw the face mesh annotations on the image.
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
          image,
          holisitcResults.face_landmarks,
          mp_holistic.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image,
            holisitcResults.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image,
            holisitcResults.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image,
            holisitcResults.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        
        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
cap.release()




def track_pose(video_path: str, segment: bool) -> None:
    mp_pose = mp.solutions.pose

    rr.log_annotation_context(
        "/",
        ClassDescription(
            info=AnnotationInfo(label="Person"),
            keypoint_annotations=[AnnotationInfo(id=lm.value, label=lm.name) for lm in mp_pose.PoseLandmark],
            keypoint_connections=mp_pose.POSE_CONNECTIONS,
        ),
    )
    # Use a separate annotation context for the segmentation mask.
    rr.log_annotation_context(
        "video/mask", [AnnotationInfo(id=0, label="Background"), AnnotationInfo(id=1, label="Person", color=(0, 0, 0))]
    )
    rr.log_view_coordinates("person", up="-Y", timeless=True)

    with closing(VideoSource(video_path)) as video_source, mp_pose.Pose(enable_segmentation=segment) as pose:
        for bgr_frame in video_source.stream_bgr():

            rgb = cv2.cvtColor(bgr_frame.data, cv2.COLOR_BGR2RGB)
            rr.set_time_seconds("time", bgr_frame.time)
            rr.set_time_sequence("frame_idx", bgr_frame.idx)
            rr.log_image("video/rgb", rgb)

            results = pose.process(rgb)
            h, w, _ = rgb.shape
            landmark_positions_2d = read_landmark_positions_2d(results, w, h)
            rr.log_points("video/pose/points", landmark_positions_2d, keypoint_ids=mp_pose.PoseLandmark)

            landmark_positions_3d = read_landmark_positions_3d(results)
            rr.log_points("person/pose/points", landmark_positions_3d, keypoint_ids=mp_pose.PoseLandmark)

            segmentation_mask = results.segmentation_mask
            if segmentation_mask is not None:
                rr.log_segmentation_image("video/mask", segmentation_mask)

@dataclass
class VideoFrame:
    data: npt.NDArray[np.uint8]
    time: float
    idx: int

class VideoSource:
    def __init__(self, path: str):
        self.capture = cv2.VideoCapture(path)

        if not self.capture.isOpened():
            logging.error("Couldn't open video at %s", path)

    def close(self) -> None:
        self.capture.release()

    def stream_bgr(self) -> Iterator[VideoFrame]:
        while self.capture.isOpened():
            idx = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            is_open, bgr = self.capture.read()
            time_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)

            if not is_open:
                break

            yield VideoFrame(data=bgr, time=time_ms * 1e-3, idx=idx)




def read_landmark_positions_2d(
    results: Any,
    image_width: int,
    image_height: int,
) -> Optional[npt.NDArray[np.float32]]:
    if results.pose_landmarks is None:
        return None
    else:
        normalized_landmarks = [results.pose_landmarks.landmark[lm] for lm in mp.solutions.pose.PoseLandmark]
        # Log points as 3d points with some scaling so they "pop out" when looked at in a 3d view
        # Negative depth in order to move them towards the camera.
        return np.array(
            [(image_width * lm.x, image_height * lm.y, -(lm.z + 1.0) * 300.0) for lm in normalized_landmarks]
        )


def read_landmark_positions_3d(
    results: Any,
) -> Optional[npt.NDArray[np.float32]]:
    if results.pose_landmarks is None:
        return None
    else:
        landmarks = [results.pose_world_landmarks.landmark[lm] for lm in mp.solutions.pose.PoseLandmark]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])

