from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


class MainApp(App):

    def build(self):
        self.icon = "Safevision.png"
        main_layout = BoxLayout(orientation="vertical")
        self.solution = TextInput(background_color="black", foreground_color="white")

        main_layout.add_widget(self.solution)

        self.args = parse_argument()
        frame_width, frame_height = args.webcam_resolution

        self.cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        model = YOLO("yolov8m.pt")

        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

        while True:
            ret, frame = cap.read()

            seresult = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)

            labels = [
                f"{model.names[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _, _
                in detections
            ]

            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

            cv2.imshow("SafeVision", frame)

            if cv2.waitKey(30) == 27:
                break


if __name__ == "__main__":
    app = MainApp()
    app.run()
