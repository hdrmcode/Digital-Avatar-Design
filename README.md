# Digital Avatar Design

A real-time, webcam-based digital avatar that mirrors a user's facial
movements and expressions. Built with Python, OpenCV, and MediaPipe Face
Mesh, the application tracks facial landmarks and drives a stylized 2D
avatar with live expression and tracking feedback.

## Overview

Digital Avatar Design connects a live webcam feed to an animated
cartoon-style avatar. The camera panel displays the user's face with a
facial landmark mesh overlay, while the avatar panel visualizes tracked
facial features.

The interface also provides live tracking information, including FPS,
blink count, eye movement, mouth movement, smile, eyebrow activity, and
head tilt.

## Features

-   **Real-time face tracking** using MediaPipe Face Mesh.
-   **468 facial landmarks and 10 iris landmarks** for detailed facial
    tracking.
-   **Live avatar visualization** with a stylized 2D character.
-   **Expression tracking** for eye movement, blinking, mouth movement,
    smiles, and eyebrows.
-   **Head movement feedback**, including head tilt and turn values.
-   **Live tracking dashboard** showing FPS and blink count.
-   **Customizable avatar appearance**, with on-screen keyboard controls
    for colours, skin tone, hair colour, hairstyle, facial hair, eyes,
    shirt, and background.
-   **Optional virtual camera support** (through the optional
    requirements file).

## Tech Stack

-   Python 3.9--3.12
-   OpenCV Contrib 4.10.0.84
-   MediaPipe 0.10.14
-   NumPy 1.26.4
-   Protobuf 4.x

## Requirements

-   Windows 10 or Windows 11
-   Python 3.9, 3.10, 3.11, or 3.12
-   A working webcam
-   A microphone is not required for the facial-tracking features shown
    in the application.

Install the pinned dependencies from `requirements.txt`. Use
**`opencv-contrib-python` only**; do not install `opencv-python`
alongside it.

## Installation

Open PowerShell or Command Prompt and move to the project folder:

``` powershell
cd "D:\Digital Avatar Design"
```

Create a virtual environment:

``` powershell
python -m venv venv
```

Upgrade pip and install the project dependencies:

``` powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install --no-compile -r requirements.txt
```

### Optional: Virtual Camera Support

If you want to use the optional virtual camera functionality and have
the `requirements-optional.txt` file in the project, install its
dependencies:

``` powershell
.\venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

## Run the Application

From the project directory, run:

``` powershell
.\venv\Scripts\python.exe avatar.py
```

Allow the application to access your webcam if Windows displays a camera
permission prompt.

## Controls

The application displays its keyboard shortcuts in the right-hand
information panel. Based on the current interface:

  Key   Action
  ----- --------------
  `A`   Auto colours
  `S`   Skin tone
  `K`   Hair colour
  `H`   Hairstyle
  `F`   Facial hair
  `E`   Eyes
  `T`   Shirt
  `B`   Background
  `M`   Mesh
  `R`   Recalibrate
  `C`   Capture
  `Q`   Quit

## Project Structure

``` text
Digital Avatar Design/
├── avatar.py
├── requirements.txt
├── requirements-optional.txt   # Optional; if included
└── README.md
```

Additional project files may be present depending on your local version.

## Troubleshooting

-   **Webcam not detected:** Check Windows camera permissions, close
    other applications that may be using the webcam, and verify that the
    correct camera is selected or configured in the application.

-   **Dependency installation errors:** Confirm that you are using
    Python 3.9--3.12 and install dependencies inside the virtual
    environment.

-   **OpenCV conflicts:** Uninstall conflicting OpenCV packages and
    reinstall only `opencv-contrib-python==4.10.0.84`:

    ``` powershell
    .\venv\Scripts\python.exe -m pip uninstall opencv-python opencv-contrib-python -y
    .\venv\Scripts\python.exe -m pip install opencv-contrib-python==4.10.0.84
    ```

-   **MediaPipe or Protobuf compatibility issues:** Keep MediaPipe at
    `0.10.14` and Protobuf below version `5`, as specified in
    `requirements.txt`.

