from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


class CamApp(App):

    def build(self):
        # Main layout components
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title = Label(text="Face Verification System", font_size=24, size_hint=(1, 0.1))
        layout.add_widget(title)
        
        # WebCam Image Display
        self.web_cam = Image(size_hint=(1, 0.6))
        layout.add_widget(self.web_cam)

        # Button Layout
        button_layout = GridLayout(cols=2, size_hint=(1, 0.1), padding=10, spacing=10)
        self.verify_button = Button(text="Verify", on_press=self.verify, size_hint_y=None, height=50)
        self.select_image_button = Button(text="Select Image from Device", on_press=self.open_filechooser, size_hint_y=None, height=50)
        button_layout.add_widget(self.verify_button)
        button_layout.add_widget(self.select_image_button)
        layout.add_widget(button_layout)

        # Verification Result
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, 0.1))
        layout.add_widget(self.verification_label)

        # Selected Image Path
        self.image_path_label = Label(text="No image selected", size_hint=(1, 0.1))
        layout.add_widget(self.image_path_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('my_model.keras', custom_objects={'L1Dist': L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        self.selected_image_path = None

        return layout

    # Run continuously to get webcam feed
    def update(self, *args):
        if not self.selected_image_path:
            # Read frame from opencv
            ret, frame = self.capture.read()
            frame = frame[120:120 + 250, 200:200 + 250, :]

            # Flip horizontally and convert image to texture
            buf = cv2.flip(frame, 0).tostring()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture

    # Open file chooser to select image from device
    def open_filechooser(self, instance):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        home_path = os.path.expanduser("~")  # Get user's home directory
        filechooser = FileChooserListView(path=home_path)
        content.add_widget(filechooser)

        button_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        select_button = Button(text="Load Image")
        cancel_button = Button(text="Cancel")
        button_layout.add_widget(select_button)
        button_layout.add_widget(cancel_button)
        content.add_widget(button_layout)

        popup = Popup(title="Select an image file", content=content, size_hint=(0.9, 0.9))
        select_button.bind(on_release=lambda x: self.load_image_from_device(filechooser.selection, popup))
        cancel_button.bind(on_release=popup.dismiss)
        popup.open()

    # Load image from device
    def load_image_from_device(self, selection, popup):
        if selection:
            self.selected_image_path = selection[0]
            self.display_selected_image()
            self.image_path_label.text = f"Selected image: {self.selected_image_path}"
        else:
            self.image_path_label.text = "No image selected"
        popup.dismiss()

    # Display selected image
    def display_selected_image(self):
        if self.selected_image_path:
            image = cv2.imread(self.selected_image_path)
            image = cv2.resize(image, (250, 250))
            buf = cv2.flip(image, 0).tostring()
            img_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture

    # Preprocess the image
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100, 100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img
    
    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.99
        verification_threshold = 0.8

        if self.selected_image_path:
            input_image_path = self.selected_image_path
        else:
            # Capture input image from our webcam
            input_image_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
            ret, frame = self.capture.read()
            frame = frame[120:120 + 250, 200:200 + 250, :]
            cv2.imwrite(input_image_path, frame)

        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(input_image_path)
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediction is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        # Set verification text 
        self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified


if __name__ == '__main__':
    CamApp().run()
