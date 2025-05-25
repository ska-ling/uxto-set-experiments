import onnx


model = onnx.load("/home/fernando/dev/utxo-experiments/model.onnx")

print("📥 Input name:", model.graph.input[0].name)
print("📤 Output name:", model.graph.output[0].name)
print("🎯 Input type:", model.graph.input[0].type.tensor_type)
