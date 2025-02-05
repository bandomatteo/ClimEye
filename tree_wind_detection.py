import cv2        
import numpy as np  

class TreeDetector:
    def __init__(self):
        self.threshold_value = 70
        self.upper_region_ratio = 0.5  

    def set_upper_region_ratio(self, ratio):
        self.upper_region_ratio = max(0.1, min(ratio, 1.0))
        
    def get_bounding_boxes(self, contours, min_area=500):
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))
        return boxes
    
    def detect_trees(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        upper_height = int(height * self.upper_region_ratio)
        upper_region = gray[0:upper_height, 0:gray.shape[1]]
        
        _, thresh = cv2.threshold(upper_region, self.threshold_value, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = self.get_bounding_boxes(contours)  
        
        return boxes, thresh
        

def main():
    cap = cv2.VideoCapture("examples/video3.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    tree_detector = TreeDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
              
        tree_boxes, thresh = tree_detector.detect_trees(frame)
              
        result = frame.copy()
        for box in tree_boxes:
            cv2.rectangle(result, (box[0], box[1]), 
                          (box[0]+box[2], box[1]+box[3]), 
                          (0, 255, 0), 2)
        
        
        cv2.imshow('Tree Detection', result)
        cv2.imshow('Threshold', thresh)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()