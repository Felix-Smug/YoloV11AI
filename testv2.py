import cv2
import numpy as np
import time
import keyboard
import pyautogui
import win32api, win32con
import math
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import bettercam

# Set up inference
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def preprocess(frame, input_shape):
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def postprocess(output, conf_threshold=0.5):
    # Example postprocess, must be adapted to your model
    # Assuming output format: [num_detections, 6] = [x1, y1, x2, y2, conf, class_id]
    detections = []
    for det in output:
        if det[4] >= conf_threshold:
            detections.append(det)
    return detections

# Load TensorRT engine
engine = load_engine("valorant.engine")
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Fire function
def keyshoot():
    win32api.keybd_event(0x4B, 0, 0, 0)
    win32api.keybd_event(0x4B, 0, win32con.KEYEVENTF_KEYUP, 0)

# Init camera
camera = bettercam.create(output_color='BGR')
screen_w, screen_h = pyautogui.size()
region_size = 150
region_left = screen_w // 2 - region_size // 2
region_top = screen_h // 2 - region_size // 2

print("Press 'P' to Stop TriggerBot")

while not keyboard.is_pressed('p'):
    frame = camera.grab(region=(region_left, region_top, region_left + region_size, region_top + region_size))

    if frame is None:
        continue

    frame_np = np.array(frame)
    img = preprocess(frame_np, engine.get_binding_shape(0))
    np.copyto(inputs[0]['host'], img.ravel())

    # Run inference
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    output = outputs[0]['host'].reshape(-1, 6)  # Adjust shape if needed
    detections = postprocess(output)

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        distance = math.hypot(box_center_x - region_size / 2, box_center_y - region_size / 2)
        if int(cls_id) == 1 and distance < 10:  # class 1 == enemy_Head
            print("enemyHead detected")
            keyshoot()
            time.sleep(0.2)
            break

print("Stopped")
