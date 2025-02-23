import cv2
import os
import time

class HybridTracker:
    def __init__(self, main_tracker, backup_tracker):
        self.main_tracker = main_tracker
        self.backup_tracker = backup_tracker
        self.frame_count = 0

    def init(self, frame, bbox):
        """Initialize both main and backup trackers."""
        self.main_tracker.init(frame, bbox)
        self.backup_tracker.init(frame, bbox)

    def update(self, frame):
        success, bbox = self.main_tracker.update(frame)
        if self.frame_count % 30 == 0:
            temp_success, temp_bbox = self.backup_tracker.update(frame)
            if temp_success:
                success = True
                bbox = temp_bbox
                self.main_tracker.init(frame, bbox)
                
        self.frame_count += 1
        return success, bbox

def GET_TRACKER():
    print('\nAvailable Trackers:')
    print('1: MIL | 2: KCF | 3: CSRT | 4: DaSiamRPN | 5: NanoTrackV2 | 6: VIT')
    choice = input('Select tracker (1-6): ')
    
    if choice == '1':
        tracker = cv2.legacy.TrackerMIL_create()
        
    elif choice == '2':
        tracker = cv2.legacy.TrackerKCF_create()
        
    elif choice == '3':
        tracker = cv2.legacy.TrackerCSRT_create()
        
    elif choice == '4':
        params = cv2.TrackerDaSiamRPN_Params()
        params.model = "models/dasiamrpn/dasiamrpn_model.onnx"
        params.kernel_cls1 = "models/dasiamrpn/dasiamrpn_kernel_cls1.onnx"
        params.kernel_r1 = "models/dasiamrpn/dasiamrpn_kernel_r1.onnx"
        params.backend = cv2.dnn.DNN_BACKEND_OPENCV
        params.target = cv2.dnn.DNN_TARGET_CPU
        tracker = cv2.TrackerDaSiamRPN_create(params)
        
    elif choice == '5':
        params = cv2.TrackerNano_Params()
        params.backbone = "models/NanoTrackV2/nanotrack_backbone_sim.onnx"
        params.neckhead = "models/NanoTrackV2/nanotrack_head_sim.onnx"
        params.backend = cv2.dnn.DNN_BACKEND_OPENCV
        params.target = cv2.dnn.DNN_TARGET_CPU
        tracker = cv2.TrackerNano_create(params)
        
    elif choice == '6':
        params = cv2.TrackerVit_Params()
        params.net = "models/VIT/object_tracking_vittrack_2023sep_int8bq.onnx"
        params.backend = cv2.dnn.DNN_BACKEND_OPENCV
        params.target = cv2.dnn.DNN_TARGET_CPU
        tracker = cv2.TrackerVit_create(params)
        
    else:
        raise ValueError("Invalid tracker choice. Please select a number between 1 and 6.")
    
    if choice in ['1', '2']:
        backup = cv2.TrackerCSRT_create()
        return HybridTracker(tracker, backup)
    
    return tracker

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture the initial frame")
    
    bbox = cv2.selectROI("Select Object", frame, False)
    cv2.destroyWindow("Select Object")
    
    tracker = GET_TRACKER()
    tracker.init(frame, bbox)

    frame_count = 0
    start_time = cv2.getTickCount()
    
    # Setup video writer
    output_dir = "/home/mohamed/Tracking/Open-CV/Output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"tracking_output_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30, (1280, 720))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = frame.copy()
        
        timer = cv2.getTickCount()
        
        success, bbox = tracker.update(processed)
        
        processing_time = (cv2.getTickCount() - timer) / cv2.getTickFrequency()
        
        x, y, w, h = [int(v) for v in bbox]

        if success:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        frame_count += 1
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        fps = frame_count / elapsed
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Latency: {processing_time*1000:.1f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Tracking", frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == 27: 
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()

