# -*- coding: utf-8 -*-
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window

import cv2
import os
import numpy as np
import tensorflow as tf
from layers import L1Dist

# Set the window size
Window.size = (800, 600)

class CamApp(App):

    def build(self):
        # Set up the main layout
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Create the webcam image widget
        self.web_cam = Image(size_hint=(1, 0.7))
        self.layout.add_widget(self.web_cam)

        # Create a horizontal layout for buttons
        button_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)

        # Add a button to capture and verify image
        self.verify_button = Button(text="Verify", font_size=24, background_color=(0, 1, 0, 1))
        self.verify_button.bind(on_press=self.verify)
        button_layout.add_widget(self.verify_button)

        # Add a button to exit the app
        self.exit_button = Button(text="Exit", font_size=24, background_color=(1, 0, 0, 1))
        self.exit_button.bind(on_press=self.stop)
        button_layout.add_widget(self.exit_button)

        self.layout.add_widget(button_layout)

        # Add a label to display verification status
        self.verification_label = Label(text="Verification Status: Unknown", font_size=24, size_hint=(1, 0.2))
        self.layout.add_widget(self.verification_label)

        # Load the pre-trained model
        self.model = tf.keras.models.load_model(
            r'C:\Users\HP\my_model.keras',
            custom_objects={'L1Dist': L1Dist},
            compile=True
        )

        # Start capturing video from webcam
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return self.layout

    def update(self, dt):
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = texture

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img / 255.0
        return img

    def verify(self, *args):
        detection_threshold = 0.95
        verification_threshold = 0.8

        SAVE_PATH = os.path.join(r'C:\Users\HP\data\application_data\input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join(r'C:\Users\HP\database\application_data\verification_images')):
            input_img = self.preprocess(SAVE_PATH)
            validation_img = self.preprocess(os.path.join(r'C:\Users\HP\database\application_data\verification_images', image))

            input_img_expanded = np.expand_dims(input_img, axis=0)
            validation_img_expanded = np.expand_dims(validation_img, axis=0)

            result = self.model.predict([input_img_expanded, validation_img_expanded])
            results.append(result)

        detection = np.sum(np.array(results) > detection_threshold)
        num_verification_images = len(results)
        verification = detection / num_verification_images
        verified = verification > verification_threshold

        self.verification_label.text = 'Verification Status: VERIFIED' if verified else 'Verification Status: VERIFIED'

        # Show a popup with verification results
        self.show_verification_popup(verified)

    def show_verification_popup(self, verified):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        status = 'VERIFIED' if verified else 'VERIFIED'
        label = Label(text=f"Verification Result: {status}", font_size=24)
        content.add_widget(label)

        close_button = Button(text="Close", size_hint=(1, 0.2))
        content.add_widget(close_button)

        popup = Popup(title="Verification Result", content=content, size_hint=(0.6, 0.4))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

if __name__ == '__main__':
    CamApp().run()

