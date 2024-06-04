PYTHON=python3 
SCRIPT=train.py

MAKEMORE=$PYTHON
MAKEMORE+=" "
MAKEMORE+=$SCRIPT

$MAKEMORE \
--type 'gru' \
--n-layer 4 \
--n-embd 64 \
--batch-size 32 \
--learning-rate 5e-4 \
--weight-decay 0.01 \
