"""
Re-exports task2.models.classifier_head -- Task 3 uses the identical
classifier head (model.fc on the shared ResNet-18), per the assignment.
"""
from task2.models.classifier_head import classify, classification_loss  # noqa: F401
