# Code used in “The emergence of integrated behavior through direct optimization for homeostasis.”

**Author note**: This repository is
However, due to recent updates to the MuJoCo environment, this code is now outdated and is not recommended for direct execution from the perspective of compatibility and work efficiency. We recommend using it for reference purposes, such as for parameters.

### Weight data
The parameters of the trained model are available at the following link.

### Environment

The following environment is required to run this experiment (execution is not recommended due to mujoco-py dependencies).

**trp-env**: https://github.com/ugo-nama-kun/trp_env_mujoco_py

**thermal-env**:  https://github.com/ugo-nama-kun/thermal_regulation_mujoco_py

### Reference
Yoshida, N., Daikoku, T., Nagai, Y., & Kuniyoshi, Y. (2024). Emergence of integrated behaviors through direct optimization for homeostasis. Neural Networks, 177, 106379.
https://www.sciencedirect.com/science/article/pii/S0893608024003034


### BibTex
```text
@article{yohida2024emergence,
	abstract = {Homeostasis is a self-regulatory process, wherein an organism maintains a specific internal physiological state. Homeostatic reinforcement learning (RL) is a framework recently proposed in computational neuroscience to explain animal behavior. Homeostatic RL organizes the behaviors of autonomous embodied agents according to the demands of the internal dynamics of their bodies, coupled with the external environment. Thus, it provides a basis for real-world autonomous agents, such as robots, to continually acquire and learn integrated behaviors for survival. However, prior studies have generally explored problems pertaining to limited size, as the agent must handle observations of such coupled dynamics. To overcome this restriction, we developed an advanced method to realize scaled-up homeostatic RL using deep RL. Furthermore, several rewards for homeostasis have been proposed in the literature. We identified that the reward definition that uses the difference in drive function yields the best results. We created two benchmark environments for homeostasis and performed a behavioral analysis. The analysis showed that the trained agents in each environment changed their behavior based on their internal physiological states. Finally, we extended our method to address vision using deep convolutional neural networks. The analysis of a trained agent revealed that it has visual saliency rooted in the survival environment and internal representations resulting from multimodal input.},
	author = {Naoto Yoshida and Tatsuya Daikoku and Yukie Nagai and Yasuo Kuniyoshi},
	doi = {https://doi.org/10.1016/j.neunet.2024.106379},
	issn = {0893-6080},
	journal = {Neural Networks},
	keywords = {Homeostasis, Deep reinforcement learning, Homeostatic reinforcement learning, Autonomous agent, Embodied agent},
	pages = {106379},
	title = {Emergence of integrated behaviors through direct optimization for homeostasis},
	url = {https://www.sciencedirect.com/science/article/pii/S0893608024003034},
	volume = {177},
	year = {2024},
	bdsk-url-1 = {https://www.sciencedirect.com/science/article/pii/S0893608024003034},
	bdsk-url-2 = {https://doi.org/10.1016/j.neunet.2024.106379}}
```
