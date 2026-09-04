#!/usr/bin/env python3
import base64
import signal
import cv2
import numpy as np
from fastapi import Response
from fastapi import Request
import torch
import torchvision
from torchvision import transforms
from nicegui import Client, app, core, run, ui
from pathlib import Path

# initialize global variables for use in coroutine functions
latest_frame = None
frame_count = 0
detected = False
tracker = None
loaded_model = None
face_cascade = None

# in case you don't have a webcam, this will provide a black placeholder image.
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')

# get the transforms required for data to be processed by MobileNetV2
weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
auto_transform = weights.transforms()
class_names = ["angry", "happy", "neutral", "sad"]


@app.post('/video/upload')
async def upload_frame(request: Request):
    global latest_frame

    data = await request.body()

    frame = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    latest_frame = frame

    return {'status': 'ok'}


def load_model(
        model: torch.nn.Module,
        path: str
):
    """
    Load a model from the given state dict
    :param path: the path to the state dict file
    :param model: a pytorch model with the same architecture as the saved model
    :return: the model with the loaded state
    """
    model_path = Path(path)
    model.load_state_dict(torch.load(f=model_path))

    return model


def read_face(image, box, detected, model):
    """
    Given the bounding box of the face, write the bounding box and the emotion to the screen. If not face detected,
    write "No face detected!"
    :param model: a loaded model that is of class nn.Module
    :param detected: indicates if a face was detected in this frame
    :param image: a 3-channel image
    :param box: bounding box in the format [x, y, width, height] (or [] if the face is not detected)
    :return: a 3-channel openCV image
    """
    if len(box) == 1:
        box = box[0]

    # display text if no face detected
    if not detected:
        cv2.putText(
            image,
            "No face detected!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

    # display a bounding box and feed the cropped image to the model
    else:
        cv2.rectangle(
            image,
            (box[0], box[1]),
            (box[0] + box[2], box[1] + box[3]),
            (255, 255, 0),
            3
        )

        # process the image into a 64x64 grayscale tensor to be fed into the model
        cropped_img = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            auto_transform
        ])
        transformed_img = data_transform(cropped_img)
        transformed_img = transformed_img.unsqueeze(0)

        # pass the processed image to the model to get a prediction
        model.eval()
        with torch.inference_mode():
            pred = model(transformed_img)
            class_pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
            idx = class_pred.item()
        cv2.putText(image, str(class_names[idx]), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA)

    # display the final image
    return image


def convert(frame: np.ndarray) -> bytes:
    """Converts a frame from OpenCV to a JPEG image.

    This is a free function (not in a class or inner-function),
    to allow run.cpu_bound to pickle it and send it to a separate process.
    """
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


def setup() -> None:
    global tracker
    global loaded_model
    global latest_frame
    global face_cascade

    # create an MNV2 model and load the saved state from the trained model
    loaded_model = torchvision.models.mobilenet_v2()
    loaded_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=4,  # same number of output units as our number of classes
                        bias=True))
    load_model(loaded_model, "models/MNV2_classifier.pth")

    # load the cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if latest_frame is not None:
        f = latest_frame.copy()
        # convert to grayscale, locate face using classifier
        frame_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

        # do not initialize the tracker until the cascade classifier has detected a face
        tracker = cv2.TrackerKCF_create()
        if len(face) != 0:
            tracker.init(f, face[0])
            detected = True

    @app.get('/video/frame')
    # Thanks to FastAPI's `app.get` it is easy to create a web route which always provides the latest image from OpenCV.
    async def grab_video_frame() -> Response:
        global frame_count
        global detected
        global tracker
        global loaded_model
        global latest_frame
        global face_cascade

        frame_count = frame_count + 1
        # return placeholder image if webcam cannot be opened or if video capture cannot be read
        if latest_frame is None:
            return placeholder

        f = latest_frame.copy()
        s = False
        box = []
        # use a tracker instead of a classifier if possible, to optimize speed
        if detected:
            s, box = tracker.update(f)

        # if there is a face detected and the tracker updates properly, process the frame
        if s:
            img = read_face(f, box, detected, loaded_model)

            # "convert" is a CPU-intensive function, so we run it in a separate process to avoid blocking the event loop and GIL.
            img = await run.cpu_bound(convert, img)
            return Response(content=img, media_type="image/jpeg")

        # use cascade classifier again if no face is lost, and every 0 frames to prevent drift
        if not s or frame_count % 50 == 0:
            tracker = cv2.TrackerKCF_create()
            frame_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
            detected = False

            # if the classifier detected a face, initialize the tracker
            if len(face) != 0:
                tracker.init(f, face[0])
                detected = True
            img = read_face(f, face, detected, loaded_model)

            # "convert" is a CPU-intensive function, so we run it in a separate process to avoid blocking the event loop and GIL.
            img = await run.cpu_bound(convert, img)
            return Response(content=img, media_type="image/jpeg")

    @ui.page('/')
    def page():

        ui.add_body_html("""
        <video id="webcam"
               autoplay
               playsinline
               style="display:none">
        </video>

        <canvas id="capture"
                style="display:none">
        </canvas>

        <script>
        async function startCamera() {

            const stream =
                await navigator.mediaDevices.getUserMedia({
                    video: true
                });

            const video =
                document.getElementById('webcam');

            const canvas =
                document.getElementById('capture');

            const ctx =
                canvas.getContext('2d');

            video.srcObject = stream;

            setInterval(async () => {

                if (video.videoWidth === 0)
                    return;

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                ctx.drawImage(
                    video,
                    0,
                    0
                );

                canvas.toBlob(async blob => {

                    if (!blob) return;

                    await fetch(
                        '/video/upload',
                        {
                            method: 'POST',
                            body: blob
                        }
                    );

                }, 'image/jpeg', 0.8);

            }, 100);
        }

        startCamera();
        </script>
        """)

        processed = ui.interactive_image(
            '/video/frame'
        ).classes('w-full')

        ui.timer(
            0.1,
            processed.force_reload
        )

    async def disconnect() -> None:
        """Disconnect all clients from current running server."""
        for client_id in Client.instances:
            await core.sio.disconnect(client_id)

    def handle_sigint(signum, frame) -> None:
        # `disconnect` is async, so it must be called from the event loop; we use `ui.timer` to do so.
        ui.timer(0.1, disconnect, once=True)
        # Delay the default handler to allow the disconnect to complete.
        ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)

    async def cleanup() -> None:
        # This prevents ugly stack traces when auto-reloading on code change,
        # because otherwise disconnected clients try to reconnect to the newly started server.
        await disconnect()

    app.on_shutdown(cleanup)
    # We also need to disconnect clients when the app is stopped with Ctrl+C,
    # because otherwise they will keep requesting images which lead to unfinished subprocesses blocking the shutdown.
    signal.signal(signal.SIGINT, handle_sigint)


# All the setup is only done when the server starts. This avoids the webcam being accessed
# by the auto-reload main process (see https://github.com/zauberzeug/nicegui/discussions/2321).
app.on_startup(setup)

ui.run(host="0.0.0.0")
