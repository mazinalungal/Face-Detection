# FaceScope — Live Face Detection

FaceScope is a browser-based webcam demo that detects faces in real time with the MediaPipe BlazeFace short-range model.

## Run locally

1. Open a terminal in this folder.
2. Start a local web server:

   ```powershell
   py -m http.server 8000
   ```

3. Visit [http://localhost:8000](http://localhost:8000) in a recent browser.
4. Select **Start detection** and allow camera access.

Camera access requires `localhost` or HTTPS. The first start downloads the MediaPipe runtime and face-detection model from their public CDNs, so an internet connection is needed. After loading, inference runs in your browser. The app does not upload or record your video. Use the snapshot button to download a still image with the detection overlay.

## Controls

- Choose an available camera while detection is running.
- Adjust the confidence threshold to filter weaker detections.
- Toggle the mirrored preview.
- Save the current frame and face boxes as a PNG.
- Select **Stop** to end the camera session and release the camera.
