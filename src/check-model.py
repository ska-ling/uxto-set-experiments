import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("utxo_hotcold_model.onnx")
input_name = session.get_inputs()[0].name

x = np.array([[
    6.3010,   # log_value
    148.0,    # total_script_size
    13513.0,  # script_efficiency
    31.0,     # block_time_proxy
    6.0,      # creation_epoch
    0.0,      # is_coinbase
    4.0,      # block_density
    0.8,      # value_percentile_in_block
    1.0,      # is_likely_change
    0.0,      # is_likely_savings
    0.0,      # coinbase_maturity_factor
    0.0,      # value_class_dust
    0.0,      # value_class_micro
    1.0,      # value_class_small
    0.0,      # value_class_medium
    0.0,      # value_class_large
    0.0       # value_class_whale
]], dtype=np.float32)

out = session.run(None, {input_name: x})
print("✅ Predicted:", out)
