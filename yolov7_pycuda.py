import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import tensorrt as trt
import numpy as np
import torch

from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.ops import nms
from utils.general import non_max_suppression as nms

import os
# Set the environment variable for CUDA lazy loading
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Yolov7Detector:
    def __init__(self, trt_engine_path, input_size, conf_th, iou_th, class_names):
        self.trt_engine_path = trt_engine_path
        self.input_size = input_size
        self.conf_th = conf_th
        self.iou_th = iou_th
        self.class_names = class_names
        self.input_width, self.input_height = input_size

        print('Loading TRT engine...')
        with open(self.trt_engine_path, 'rb') as f:
            engine_data = f.read()

        print('Building TRT engine...')
        self.trt_logger = trt.Logger()
        trt_runtime = trt.Runtime(self.trt_logger)
        self.engine = trt_runtime.deserialize_cuda_engine(engine_data)
        if not isinstance(self.engine, trt.ICudaEngine):
            raise TypeError("self.engine is not an instance of trt.ICudaEngine")
        if not os.path.isfile(self.trt_engine_path):
            raise FileNotFoundError(f"{self.trt_engine_path} does not exist")
        
        self.context = self.engine.create_execution_context()
        self.batch_size = 1
        print('Done.')
        print(input_size)
        #self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine, self.batch_size)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        print('Done.1')

    #def allocate_buffers(engine, batch_size):
    # def allocate_buffers(self):
    #     inputs = []
    #     outputs = []
    #     bindings = []
    #     stream = cuda.Stream()

    #     for binding in self.engine:
    #         shape = self.engine.get_tensor_shape(binding)
    #         dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

    #         # Modify the shape to include the batch size
    #         shape = (self.batch_size,) + shape[1:]

    #         # Calculate the size of the buffer
    #         size = trt.volume(shape) * self.batch_size * dtype.itemsize

    #         # Allocate host and device buffers
    #         host_mem = cuda.pagelocked_empty(size, dtype)
    #         device_mem = cuda.mem_alloc(host_mem.nbytes)

    #         # Append the inputs/outputs/bindings
    #         if self.engine.binding_is_input(binding):
    #             inputs.append({'host': host_mem, 'device': device_mem})
    #             bindings.append(int(device_mem))
    #         else:
    #             outputs.append({'host': host_mem, 'device': device_mem})
    #             bindings.append(int(device_mem))

    #     return inputs, outputs, bindings, stream
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        print('allocate_buffers starting...')

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

            # if isinstance(shape, tuple):
            if shape is not None:
                size = trt.volume(shape) * self.batch_size * np.dtype(dtype).itemsize

                # Allocate device memory for the binding
                mem = cuda.mem_alloc(size)

                # Append the device memory allocation to the list of bindings
                bindings.append(int(mem))

                # Append an empty NumPy array to either the inputs or outputs list, depending on whether the binding is an input or output
                if self.engine.binding_is_input(binding):
                    host_array = np.empty((self.batch_size,) + shape[1:], dtype=dtype)
                    inputs.append({'host': host_array,
                                   'device': cuda.mem_alloc(host_array.nbytes)})
                else:
                    host_array = np.empty((self.batch_size,) + shape[1:], dtype=dtype)
                    outputs.append({'host': host_array,
                                   'device': cuda.mem_alloc(host_array.nbytes)})
            else:
                print(f"Invalid shape for binding '{binding}': {shape}")

        return inputs, outputs, bindings, stream

    print('Loading TRT engine done!')
    def preprocess_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = image.resize((self.input_width, self.input_height))

        # Preprocess the image
        transform = ToTensor()
        image = transform(image)
        image = image.unsqueeze(0).cuda()

        return image

    def do_inference(self, image):

        np.copyto(self.inputs[0]['host'], np.array(image.cpu()))
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)


        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

    def postprocess_output(self, output):
        # Modify this method based on the output tensor format of your model
        detections = output[0]
        mask = detections[:, 4] >= self.conf_th
        detections = detections[mask]

        boxes = detections[:, :4]
        confidences = detections[:, 4]
        class_ids = detections[:, 5]

        if class_ids.shape[0] > 0:
            # Apply non-maximum suppression
            keep = nms(boxes, confidences, self.iou_th)
            boxes = boxes[keep].cpu().numpy()
            confidences = confidences[keep].cpu().numpy()
            class_ids = class_ids[keep].cpu().numpy()

            # Convert the bounding boxes to x1, y1, x2, y2 format
            boxes[:, [0, 2]] *= self.input_width
            boxes[:, [1, 3]] *= self.input_height

            # Convert class indices to class names
            class_names = [self.class_names[int(class_id)] for class_id in class_ids]

            return boxes, confidences, class_ids, class_names
        return [], [], [], []


    def detect_objects_realtime(self):
        cap = cv2.VideoCapture('rtsp://admin:Admin1357@192.168.0.190:554/profile2/media.smp')  # read from camera

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = self.preprocess_image(frame)
            self.do_inference(image)
            output = self.outputs[0]['host']
            boxes, confidences, class_ids, class_names = self.postprocess_output(output)
            
            if len(class_ids) > 0:
                # Draw the bounding boxes and labels on the frame
                for box, confidence, class_id, class_name in zip(boxes, confidences, class_ids, class_names):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f'{class_name}: {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            print(frame.shape)

            # Display the frame
            cv2.imshow('Object Detection', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()



# Example usage

if __name__ == "__main__":
    # Define the paths and parameters
    trt_engine_path = '/home/tcom/yolov7/yolov7.trt'
    input_size = (640, 640)
    conf_th = 0.5
    iou_th = 0.5
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

    print('Initializing the detector...')

    # Create an instance of Yolov7Detector
    detector = Yolov7Detector(trt_engine_path, input_size, conf_th, iou_th, class_names)

    # Start real-time object detection
    detector.detect_objects_realtime()
