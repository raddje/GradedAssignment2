# Snake Reinforcement Learning (DQN – PyTorch, GA02)

This project implements a Deep Q-Network (DQN) agent to play the Snake game using the
environment and utilities provided in the assignment.  
All TensorFlow components were removed and replaced with a PyTorch implementation.

The final submitted code runs **inside the Conda environment from GA01 (uitnn)** 

---

# 🧩 1. How to Activate the Conda Environment

```
conda activate uitnn
cd *folder where the project is located*
```

---

# 🏋️ 2. How to Train the Agent

### Basic training

```
python training.py 
```

### Custom run

```
python training.py --episodes 55000 --games 64 --gamma 0.99     --eps-start 1.0 --eps-end 0.05 --eps-decay 0.97
```

---

## ⚙️ Training Parameters

| Argument | Meaning | Default |
|---------|---------|---------|
| --episodes | Training iterations | 50000 |
| --games | Parallel games | 64 |
| --eval-games | Eval games | 10 |
| --gamma | Discount factor | 0.99 |
| --eps-start | Initial epsilon | 1.0 |
| --eps-end | Minimum epsilon | 0.01 |
| --eps-decay | Log-step decay | 0.97 |
| --batch | DQN batch size | 64 |
---

# 🎮 3. Evaluate & Visualize the Final Policy

```
python eval_and_visualize.py
```

### Optional:

```
python eval_and_visualize.py --iter 15000
python eval_and_visualize.py --no-video
```

---

# 🚀 4. Workflow Summary

Train:

```
python training.py 
```

Evaluate best:

```
python eval_and_visualize.py
```

Replay a specific version:

```
python eval_and_visualize.py --iter 15000
```

---

# 📌 Additional dependencies
For recording final result:

imageio
imageio-ffmpeg
opencv-python
