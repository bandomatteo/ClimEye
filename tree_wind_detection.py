import cv2        
import numpy as np  
from collections import deque

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
    
class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None
        self.movement_history = deque(maxlen=5)
        self.resize_dim = 50  
    
    def set_resize_dim(self, dim):     
        self.resize_dim = max(10, min(dim, 200))

    def calculate_motion(self, gray_frame, roi):
        x, y, w, h = roi
            
        roi_frame = cv2.resize(gray_frame[y:y+h, x:x+w], 
                               (self.resize_dim, self.resize_dim))
        
        # Inizializzazione del primo frame
        if self.prev_gray is None or self.prev_gray.shape != roi_frame.shape:
            self.prev_gray = roi_frame
            return 0.0
        
        # Calcola la differenza assoluta tra frame consecutivi
        flow = cv2.absdiff(self.prev_gray, roi_frame)
        
        cv2.imshow('Flow', flow)
        
        # Calcola il movimento medio
        movement = np.mean(flow)
        
        # Aggiungi alla storia dei movimenti
        self.movement_history.append(movement)
        
        # Aggiorna il frame precedente
        self.prev_gray = roi_frame
        
        return movement

    def is_windy(self, threshold=45):
        
        if len(self.movement_history) == 0:
            return False  
    
        avg_movement = np.mean(self.movement_history)
        
        print(f"Average Movemetn: {avg_movement} | Threshold: {threshold}")
        
        return avg_movement > threshold
        

def main():
    cap = cv2.VideoCapture("examples/video2.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    tree_detector = TreeDetector()
    motion_analyzer = MotionAnalyzer()
    
    cv2.namedWindow('Parameters')
    def nothing(x):
        pass
    
    cv2.createTrackbar('Upper Region %', 'Parameters', 50, 100, nothing)
    cv2.createTrackbar('Wind Threshold', 'Parameters', 45, 100, nothing)
    

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        upper_region_ratio = cv2.getTrackbarPos('Upper Region %', 'Parameters') / 100.0
        wind_threshold = cv2.getTrackbarPos('Wind Threshold', 'Parameters')
        
        #motion_analyzer.set_resize_dim(resize_dim)
        tree_detector.set_upper_region_ratio(upper_region_ratio)
              
        tree_boxes, thresh = tree_detector.detect_trees(frame)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           
        result = frame.copy()
        for box in tree_boxes:
            motion_score = motion_analyzer.calculate_motion(gray_frame, box)
            is_windy = motion_analyzer.is_windy(threshold=wind_threshold)
            
            if not is_windy:
                color = (0, 255, 0)  # Verde per alberi fermi
            else:
                color = (0, 0, 255)  # Rosso per alberi mossi dal vento
            # Rectangle (img , (x, y), (x + w, y + h), color in BGR, thickness) 
            cv2.rectangle(result, (box[0], box[1]), 
                          (box[0]+box[2], box[1]+box[3]), 
                          color, 2)
        
        
        cv2.imshow('Tree Detection', result)
        cv2.imshow('Threshold', thresh)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()