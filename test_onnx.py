import onnx

model = onnx.load("onnx_model/yolo_tiny.onnx")
onnx.checker.check_model(model, True)
