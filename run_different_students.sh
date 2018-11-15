set -eux

TEACHER_CHECKPOINT_DIR="results/Pong-v0_greyscale_mse_2018-11-06-13-43-39/teacher/model.weights/"

## Softmax sharpening

# MSE of action probabilities with softmax sharpening
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_mse_prob_softmax_0.01 $TEACHER_CHECKPOINT_DIR mse_prob softmax_tau -stqt 0.01


## No softmax

# KL without softmax sharpening/softening
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_kl $TEACHER_CHECKPOINT_DIR kl none


## Softmax softening
# KL with softmax softening
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_kl_softmax_2 $TEACHER_CHECKPOINT_DIR kl softmax_tau -stqt 2

# MSE of action probabilities with softmax softening
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_mse_prob_softmax_2 $TEACHER_CHECKPOINT_DIR mse_prob softmax_tau -stqt 2


## DON'T DO THESE

# MSE of Q values with softmax sharpening
# This will do the MSE between the teacher Q values divided by tau and the student
# Q values, which is kinda the same as just doing the MSE
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_mse_qval_softmax_0.01 $TEACHER_CHECKPOINT_DIR mse_qval softmax_tau -stqt 0.01

# MSE of action probabilities with or without softmax sharpening/softening
# This will just make the student learn through the softmax function, which isn't
# necessary unless we want the student's Q values to be of the same scale
# relative to teach other but of a different absolute scale (since only the relative
# scale affects the action probabilities due to the softmax).
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_mse_prob $TEACHER_CHECKPOINT_DIR mse_prob