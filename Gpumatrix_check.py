import cv2

def check_frame_type(frame):
    if isinstance(frame, cv2.cuda_GpuMat):
        print("Frame type: Cuda matrix (cv2.cuda_GpuMat)")
    elif isinstance(frame, cv2.Mat):
        print("Frame type: CPU matrix (cv2.Mat)")
    else:
        print("Unknown frame type")

# Example usage
frame_gpu = cv2.cuda_GpuMat()
frame_cpu = cv2.imread("image.jpg")

check_frame_type(frame_gpu)  # Output: Frame type: Cuda matrix (cv2.cuda_GpuMat)
check_frame_type(frame_cpu)  # Output: Frame type: CPU matrix (cv2.Mat)
