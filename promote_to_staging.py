from mlflow.tracking import MlflowClient

client = MlflowClient()

MODEL_NAME = "IrisClassifier"
VERSION = 1

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=VERSION,
    stage="Staging"
)

# Confirm the change
model_version = client.get_model_version(MODEL_NAME, VERSION)
print(f"Model: {model_version.name}")
print(f"Version: {model_version.version}")
print(f"Current stage: {model_version.current_stage}")