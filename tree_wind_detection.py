import cv2        
import numpy as np  

def main():
       
    cap = cv2.VideoCapture("examples/video3.mp4")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Video ended. I am gonna restard =)")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        cv2.imshow('Video', frame)
              
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()