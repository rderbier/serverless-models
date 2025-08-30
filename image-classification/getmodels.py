from transformers import AutoImageProcessor, AutoModelForImageClassification
modelPath = "./model"
model_name = 'google/vit-base-patch16-224'
# Load model directly


processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.save(modelPath)
processor.save(modelPath)

