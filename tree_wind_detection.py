import cv2        
import numpy as np  
from collections import deque

class TreeDetector:
    def __init__(self):
        self.threshold_value = 70
        self.upper_region_ratio = 0.5  

    def set_upper_region_ratio(self, ratio): 
        self.upper_region_ratio = max(0.1, min(ratio, 1.0))
        
    def get_bounding_boxes(self, contours, min_area=800):
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
        
        blurred = cv2.GaussianBlur(upper_region, (3, 3), 0)
        
        _, thresh = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = self.get_bounding_boxes(contours)  
        
        return boxes, thresh
    
class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = {}
        self.movement_histories = {}
        self.resize_dim = 50
        self.tree_counter = 0  
        self.last_known_positions = {} # Centers of the trees
        self.max_distance = 50  # Just to make sure it's the same tree
        
    def _get_center(self, roi):
        x, y, w, h = roi
        return (x + w//2, y + h//2)
    
    def _get_closest_tree_id(self, current_center):
        closest_id = None
        min_distance = float('inf')
        
        for tree_id, last_center in self.last_known_positions.items():
            distance = np.sqrt((current_center[0] - last_center[0])**2 + 
                             (current_center[1] - last_center[1])**2)
            if distance < min_distance and distance < self.max_distance:
                min_distance = distance
                closest_id = tree_id
                
        return closest_id
    
    def _get_tree_id(self, roi):
        current_center = self._get_center(roi)
        
        closest_id = self._get_closest_tree_id(current_center)
        
        if closest_id is None:
            closest_id = f"tree_{self.tree_counter}"
            self.tree_counter += 1
            
        self.last_known_positions[closest_id] = current_center
        
        return closest_id
    
    #def set_resize_dim(self, dim):     
        #self.resize_dim = max(10, min(dim, 200))

    def calculate_motion(self, gray_frame, roi):
        tree_id = self._get_tree_id(roi)
        x, y, w, h = roi
    
        if tree_id not in self.movement_histories:
            self.movement_histories[tree_id] = deque(maxlen=5)
            
        roi_frame = cv2.resize(gray_frame[y:y+h, x:x+w], 
                             (self.resize_dim, self.resize_dim))
         
        if tree_id not in self.prev_gray or self.prev_gray[tree_id].shape != roi_frame.shape:
            self.prev_gray[tree_id] = roi_frame
            return 0.0, tree_id
        
        flow = cv2.absdiff(self.prev_gray[tree_id], roi_frame)
        
        # Flows for debuggig
        #cv2.imshow(f'Flow_{tree_id}', flow)
             
        movement = np.mean(flow)
        
        self.movement_histories[tree_id].append(movement)
        print(f"Tree {tree_id} - Movement: {movement}")
        
        self.prev_gray[tree_id] = roi_frame
        
        return movement, tree_id

    def is_windy(self, tree_id, threshold=2):
        if tree_id not in self.movement_histories or len(self.movement_histories[tree_id]) == 0:
            return False
        
        avg_movement = np.mean(self.movement_histories[tree_id])
        print(f"Tree {tree_id} - Average Movement: {avg_movement} | Threshold: {threshold}")
        
        return avg_movement > threshold

    def get_average_movement(self):
        if not self.movement_histories:
            return 0.0
        
        total_movement = 0
        count = 0
        
        for tree_movements in self.movement_histories.values():
            if tree_movements:  # Check if the deque is not empty
                total_movement += sum(tree_movements) / len(tree_movements)
                count += 1
        
        return total_movement / count if count > 0 else 0.0

def main():
    cap = cv2.VideoCapture("examples/windy1.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    tree_detector = TreeDetector()
    motion_analyzer = MotionAnalyzer()
    
    cv2.namedWindow('Parameters')
    def nothing(x):
        pass
    
    cv2.createTrackbar('Upper Region %', 'Parameters', 50, 100, nothing)
    cv2.createTrackbar('Wind Threshold', 'Parameters', 2, 100, nothing)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        upper_region_ratio = cv2.getTrackbarPos('Upper Region %', 'Parameters') / 100.0
        wind_threshold = cv2.getTrackbarPos('Wind Threshold', 'Parameters')
        
        tree_detector.set_upper_region_ratio(upper_region_ratio)
           
        tree_boxes, thresh = tree_detector.detect_trees(frame)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        result = frame.copy()
        for box in tree_boxes:
            motion_score, tree_id = motion_analyzer.calculate_motion(gray_frame, box)
            is_windy = motion_analyzer.is_windy(tree_id, threshold=wind_threshold)
            
            color = (0, 0, 255) if is_windy else (0, 255, 0)
                
            cv2.rectangle(result, (box[0], box[1]), 
                         (box[0]+box[2], box[1]+box[3]), 
                         color, 2)
            
            text = f"Tree #{tree_id.split('_')[1]}"
            cv2.putText(result, text, (box[0], box[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
   
        avg_movement = motion_analyzer.get_average_movement()
        condition = "WINDY" if avg_movement > wind_threshold else "CALM"
        condition_color = (0, 0, 255) if condition == "WINDY" else (0, 255, 0)

        cv2.putText(result, f"Condition: {condition}", 
                    (10, result.shape[0] - 20),  # Bottom left
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7,  
                    condition_color,
                    2)  
        
        cv2.imshow('Tree Detection', result)
        cv2.imshow('Threshold', thresh)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()