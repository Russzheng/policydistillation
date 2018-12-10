set -eux

TEACHER_CHECKPOINT_DIR="./results/PongNoFrameskip-v4_greyscale_huber_teacher/teacher_PongNoFrameskip-v4_greyscale_huber_teacher/model.weights/"
TEACHER_NAME="PongNoFrameskip-v4_greyscale_huber"

## Softmax sharpening

# MSE of action probabilities with softmax sharpening
# DONE: max evaluated reward: 11.32 +/- 0.50
# python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_mse_prob_softmax_0.01 $TEACHER_CHECKPOINT_DIR mse_prob softmax_tau -stqt 0.01
# debugging new multi-teacher format
python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_huber_teacher -sl mse_prob -ptq softmax_tau -ctq none -ep greedy -stqt 0.01 -tcd $TEACHER_CHECKPOINT_DIR -tcn $TEACHER_NAME