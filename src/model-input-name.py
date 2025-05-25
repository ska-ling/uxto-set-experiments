import onnx

model = onnx.load("/home/fernando/dev/utxo-experiments/model.onnx")
input_name = model.graph.input[0].name
print(f"📥 Nombre del input: {input_name}")
