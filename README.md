# pytorch-a3c

This is a PyTorch implementation of Asynchronous Advantage Actor Critic (A3C) from ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783v1.pdf).

This implementation is inspired by [Universe Starter Agent](https://github.com/openai/universe-starter-agent).

## Contibutions

Contributions are very welcome. If you know how to make this code better, don't hesitate to send a pull request.

## Usage
```
python main.py --env-name "PongDeterministic-v3" --num-processes 16
```

## Todo

- [ ] Deterministic evaluation in a separate thread
