from collections import OrderedDict
import numpy as np

config = dict()

training_year = 2021
config["training_year"] = training_year
prediction_year = 2021
config["prediction_year"] = prediction_year

config["dataset_path"] = "dataset_path"

config["preprocessed_data_dir"] = config["dataset_path"] + "/preprocessed_training_data/" + str(training_year)

config["h_input"] = 240
config["w_input"] = 240
config["d_input"] = 155

config["num_modalities"] = 4

config["num_mod_gt"] = 5

config["num_classes"] = 4

config["network_dims"] = "2D"
config["network_view"] = "Axial"

config["h_patch"] = 128
config["w_patch"] = 128
config["d_patch"] = 128

config["extraction_type"] = "random_locations"

config["seg_model_name"] = "2D ESA-Net"

config["base_channels"] = 32

config["num_levels"] = 4

config["Conv_type"] = "Depthwise separable convolution"

config["norm"] = "Batch Normalization"
config["batch_size"] = 8

config["activation"] = "relu"

config["downsampling"] = "Max pooling"

config["upsampling"] = "Bilinear upsampling"

config["attn_module_type"] = "SA"

config["skip_connection"] = "Concatenate"

config["init_lr"] = 1e-4

config["weight_decay"] = 1e-5

config["seg_training_model_name"] = "seg_training_model"
config["seg_prediction_model_name"] = "seg_prediction_model"

config["num_epochs"] = 300
config["save_only_best_model"] = False
config["save_period"] = 1
config["validation_steps"] = 2

config["gpu_cpu"] = "GPU"
config["gpu_mem"] = "memory growth"
config["gpu_mem_limit"] = 2048
config["mixed_precision_training"] = True

config["pred_batch_size"] = 32
config["pred_model_type"] = "native TF FP32"

config["prediction_type"] = "Individual"

config["ensemble_name"] = "Ensemble_1"

config["tta"] = False

config["postprocessing"] = False

config["num_survival_classes"] = 3

config["feature_type"] = "Deep features"

config["h_resized"] = 64
config["w_resized"] = 64
config["d_resized"] = 64

config["imputer_name"] = "imputer.joblib"
config["scaler_name"] = "scaler.joblib"

config["feature_selection"] = "Pearson correlation coefficient"
config["feature_selector_name"] = "feature_selector.joblib"

config["survival_model"] = "XGBoost regression"

config["xgb_model_name"] = "surv_xgb_model.model"
