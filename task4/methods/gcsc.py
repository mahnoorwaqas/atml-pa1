"""
Step 3: GCSC -- exactly the Vanilla recipe, with RandAugment(num_ops=2,
magnitude=9) inserted after crop+flip, before ToTensor/Normalize. Reuses
vanilla.py's build_model/train_cifar_classifier UNCHANGED -- the only
difference is the transform passed in by train.py, not any code here.
"""
from task4.methods.vanilla import build_model, train_cifar_classifier, evaluate_accuracy  # noqa: F401
