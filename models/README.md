# Trained model artifacts are stored here.
# They are excluded from Git (see .gitignore) but persisted via Docker volume.
#
# Files that will appear here as you complete each phase:
#
#   models/tier1_svm.pkl          ← after: python training/train_tier1.py
#   models/teacher_bert/          ← after: python -m ml.slm.01_teacher.train_teacher
#   models/student_distilled/     ← after: python -m ml.slm.02_distillation.distill_train
#   models/student_pruned/        ← after: python -m ml.slm.03_pruning.run_pruning
#   models/student_quantized.pt   ← after: python -m ml.slm.04_quantization.quantize_dynamic
#   models/student_tokenizer/     ← saved alongside quantized model
