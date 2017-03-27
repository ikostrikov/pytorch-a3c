# pytorch-a3c

This is a PyTorch implementation of Asynchronous Advantage Actor Critic (A3C) from ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783v1.pdf).

This implementation is inspired by [Universe Starter Agent](https://github.com/openai/universe-starter-agent).
In contrast to the starter agent, it uses an optimizer with shared statistics as in the original paper.

## Contibutions

Contributions are very welcome. If you know how to make this code better, don't hesitate to send a pull request.

## Usage
```
OMP_NUM_THREADS=1 python main.py --env-name "PongDeterministic-v3" --num-processes 16
```

This code runs evaluation in a separate thread in addition to 16 processes.

Note:
Install most recent nightly build (version '0.1.10+2fd4d08' or later) of PyTorch via this command to prevent memory leaks:
`
pip install git+https://github.com/pytorch/pytorch
`

## Results

With 16 processes it converges for PongDeterministic-v3 in 15 minutes.
![PongDeterministic-v3](images/PongReward.png)

For BreakoutDeterministic-v3 it takes more than several hours.
