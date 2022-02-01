from sys import byteorder

import tflite_runtime.interpreter as tflite
import picamera
import cv2
import numpy as np
from typing import *
import io
import os
import re


class CameraGrabber:
    def __init__(self, save_dir=""):
        self._cam_running: bool = False
        self._save_dir: str = save_dir

    def go(self):
        self._cam_running = True
        while self._cam_running:
            stream = io.BytesIO()
            with picamera.PiCamera() as camera:
                camera.capture(stream, format='bmp')  # Grab frame from camera

            buff = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(buff, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # The TFLite model expects a B/W image

            face_cascade = cv2.CascadeClassifier(PATH_TO_HAARCASCADE)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces):  # if a face has been detected
                for (x, y, w, h) in faces:  # for each position and size of face, hand it to the Recogniser
                    print("found {} faces".format(len(faces)))
                    gray = cv2.resize(gray[y:y + h, x:x + w], (92, 112))
                    recog = Recogniser(image=gray, model_file=PATH_TO_TFLITE, controller_dir=PATH_TO_KNOWN_FACES)
                    recog.flow()

    def stop(self):
        self._cam_running = False

    def newFaceMode(self):
        assert self._save_dir[-4:] == ".pgm", "save dir must use .pgm extension."
        face_found = False
        while not face_found:
            stream = io.BytesIO()
            with picamera.PiCamera() as camera:
                camera.capture(stream, format='bmp')

            buff = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(buff, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(PATH_TO_HAARCASCADE)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 1:
                x, y, w, h = faces[0]
                resized_to_face = cv2.resize(gray[y:y + h, x:x + w], (92, 112))
                cv2.imwrite(self._save_dir, resized_to_face)
                face_found = True
                print("face added to ::: {}".format(self._save_dir))


class KnownFacesController:
    def __init__(self, root_dir: str):
        self._root_dir = root_dir

    def getFaceDirs(self) -> Dict:
        folders: List = [f[0] for f in os.walk(self._root_dir) if f[0] != self._root_dir]
        files = {folder: os.listdir(folder) for folder in folders}
        return files


class Recogniser:
    def __init__(self, image, model_file, controller_dir):
        self._image = image
        self._interpreter = tflite.Interpreter(model_path=model_file)
        self._predictions: Dict = {}
        self._controller = KnownFacesController(controller_dir)

    def flow(self):
        folders: Dict = self._controller.getFaceDirs()
        for folder, files in folders.items():
            distances = []
            for file in files:
                data = self.prepareData(folder + "/" + file)
                distances.append(self.makePrediction(data).item(0))
            self._predictions[re.search(r"\w+\Z", folder).group()] = distances
        self.conclude()

    def conclude(self):
        print(self._predictions)

    @staticmethod
    def convertPGMtoNumpy(img_dir: str):
        with open(img_dir, "rb") as file:
            buffer = file.read()
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        return np.frombuffer(buffer,
                             dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                             count=int(width) * int(height),
                             offset=len(header)
                             ).reshape((int(height), int(width)))

    def prepareData(self, img_dir: str):
        im1 = np.zeros([1, 112, 92], dtype=np.float32)
        im1[0] = self._image
        im1 = np.expand_dims(im1, axis=3)

        im2 = np.zeros([1, 112, 92], dtype=np.float32)
        im2[0] = self.convertPGMtoNumpy(img_dir)
        im2 = np.expand_dims(im2, axis=3)
        return [im1, im2]

    def makePrediction(self, input_data: List):
        self._interpreter.allocate_tensors()
        input_details, output_details = (self._interpreter.get_input_details(), self._interpreter.get_output_details())

        self._interpreter.set_tensor(input_details[0]['index'], input_data[0])
        self._interpreter.set_tensor(input_details[1]['index'], input_data[1])

        self._interpreter.invoke()
        return self._interpreter.get_tensor(output_details[0]['index'])


def Main():
    if int(input("1 for New, _ for live recognition")) == 1:
        cam = CameraGrabber(save_dir=SAVE_PATH)
        cam.newFaceMode()
    else:
        maincam = CameraGrabber()
        maincam.go()


if __name__ == "__main__":
    Main()
