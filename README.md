# ClimEye
![logo](https://i.ibb.co/B5xs3ygP/Risorsa-14x.png)

A computer vision system that detects trees in video footage and analyzes their movement to determine wind conditions. The system uses OpenCV to identify trees and track their motion, providing real-time feedback about whether conditions are windy or calm.

![Tree Detection Demo](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2h6NnZ0c2dhbW9oMnMwNGxid2NqamprcXB1bjN2NHVqNmludG5maCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/BXXUjDn1CCLqoxrNSm/giphy.gif)

## Features

The system provides several key capabilities:

- Real-time tree detection using contour analysis
- Individual tree tracking and movement analysis
- Wind condition classification (Windy vs Calm)
- Interactive parameter adjustment through trackbars
- Visual feedback with bounding boxes and status indicators

## How It Works

The system operates using two main components:

### Tree Detector

The TreeDetector class handles the identification of trees in the video frame:

- Converts input frames to grayscale
- Applies Gaussian blur for noise reduction
- Uses threshold-based segmentation to identify potential trees
- Generates bounding boxes around detected trees
- Allows adjustment of the upper region ratio for detection

### Motion Analyzer

The MotionAnalyzer class tracks and analyzes tree movement:

- Maintains movement history for each detected tree
- Calculates motion using frame differences
- Provides individual tree movement scores
- Determines overall wind conditions based on average movement
- Tracks trees across frames using position-based matching

![Motion Analysis Example](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExYWh4bTEweHZxdG9pNHRwbnRzaGxuNWYzZXBlYTYzZnJ6eDEweWx0NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Wo0O5dCzSayKtPJ5Zq/giphy.gif)

## Installation

```bash
# Clone the repository
git clone https://github.com/bandomatteo/ClimEye.git

# Navigate to the project directory
cd ClimEye

# Install required dependencies
pip install -r requirements.txt
```

## Dependencies

- OpenCV (cv2)
- NumPy
- Python 3.7+

## Usage

Run the main script to start the detection system:

```bash
python tree_wind_detection.py
```

### Parameter Adjustment

The system provides two adjustable parameters through the interface:

1. Upper Region %: Controls the vertical portion of the frame used for tree detection
2. Wind Threshold: Adjusts the sensitivity of wind detection

![Parameter Controls](https://i.ibb.co/x8GC9DwC/image.png)

### Controls

- Press 'q' to quit the application
- Use trackbars to adjust detection parameters in real-time

## Output Windows

The system displays three windows:

1. Tree Detection: Shows the processed frame with bounding boxes and wind status
2. Threshold: Displays the binary threshold image used for detection
3. Frame: Shows the original input video


## Notes

- The system performs best with stationary camera footage
- Optimal performance requires good lighting conditions
- Trees should be clearly visible against the background
- Performance may vary based on video quality and resolution

## Video Sources
- [calm.mp4](https://www.youtube.com/)
- [windy1.mp4](https://stock.adobe.com/it/video)
- [windy1.mp4](https://stock.adobe.com/it/video)
- [windy3.mp4](https://www.storyblocks.com/)

