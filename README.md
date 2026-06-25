# 📈 Exploring Pose-Guided Imitation Learning for Robotic Precise Insertion

[[Paper]](https://www.arxiv.org/abs/2505.09424) 



https://github.com/user-attachments/assets/d62967ce-b3e9-4fcd-8121-99bf69d4d750



## 🛫 Getting Started

### 💻 Installation

Please follow the instructions to install the conda environments, as well as the real robot environments. We recommend using CUDA 11.8 during installations to avoid compatibility issues. 
Also, remember to adjust the Hand-Eye-Calibration in `eval_ros_Pose.py` and `eval_ros_RPDP.py` according to your own environment.

1. Create a new conda environment and activate the environment.
    ```bash
    conda create -n PoseInsert python=3.8
    conda activate PoseInsert
    ```

2. Manually install cudatoolkit, then install necessary dependencies.
    ```bash
    pip install -r requirements.txt
    ```

3. Install [FoundationPose](https://github.com/NVlabs/FoundationPose) (Docker, CUDA 12.1 for RTX 40-series):
    ```bash
    # image (China mirror if docker.io fails)
    docker pull docker.m.daocloud.io/shingarey/foundationpose_custom_cuda121:latest
    docker tag docker.m.daocloud.io/shingarey/foundationpose_custom_cuda121:latest foundationpose:latest

    # first launch + one-time build (inside container)
    cd FoundationPose/docker && bash run_container.sh
    bash build_all.sh

    # weights (wget from HF mirror; put under FoundationPose/weights/)
    mkdir -p FoundationPose/weights/{2023-10-28-18-33-37,2024-01-11-20-02-45}
    wget -c https://hf-mirror.com/gpue/foundationpose-weights/resolve/main/2023-10-28-18-33-37/model_best.pth -O FoundationPose/weights/2023-10-28-18-33-37/model_best.pth
    wget -c https://hf-mirror.com/gpue/foundationpose-weights/resolve/main/2024-01-11-20-02-45/model_best.pth -O FoundationPose/weights/2024-01-11-20-02-45/model_best.pth

    # demo: download mustard0.zip from FP readme → unzip to FoundationPose/demo_data/

    # run demo with live window
    xhost +local:root
    docker exec -it -e DISPLAY=$DISPLAY -e XAUTHORITY=$HOME/.Xauthority -e QT_X11_NO_MITSHM=1 foundationpose bash -c "cd $HOME/Documents/FoundationPose && python run_demo.py"
    ```

4. **(Optional) Quick FP peg test** — record aligned RealSense RGB-D on the host (not in Docker), then run FP `run_demo.py`:
    ```bash
    pip install pyrealsense2
    # Space: start/stop recording, q: quit
    python collect_data/record_realsense_foundationpose.py
    # default output: ~/Documents/FoundationPose/demo_data/peg_test/
    ```

    **First-frame mask (required once per scene):** FoundationPose only needs a mask on the **first RGB frame** to initialize tracking. The file must live in `masks/` and use the **same filename** as the first file in `rgb/` (sorted alphabetically; with the recorder that is usually the earliest timestamp).

    ```
    peg_test/
    ├── rgb/1782369494161.png      ← first frame
    ├── depth/1782369494161.png
    ├── masks/1782369494161.png    ← you create this (object=white, background=black)
    └── cam_K.txt
    ```

    Easiest: paint it with the helper script (run on host, needs a display).
    Default scene is `peg_test`; use `--scene_dir` for hole or other scenes:
    ```bash
    # peg mask (default)
    python collect_data/paint_first_frame_mask.py
    # or set initial brush radius: --brush 20

    # hole mask (separate scene dir — do not overwrite peg mask in peg_test/)
    python collect_data/paint_first_frame_mask.py \
      --scene_dir ~/Documents/FoundationPose/demo_data/hole_test \
      --brush 20
    ```
    Controls: left=paint, right=erase, `[`/`]`=brush size at runtime, z=undo, s=save, r=reset, q=quit

    **Track peg** (same recorded video, `peg_test/`):
    ```bash
    docker exec -it -e DISPLAY=$DISPLAY -e XAUTHORITY=$HOME/.Xauthority -e QT_X11_NO_MITSHM=1 foundationpose bash -c \
      "python $HOME/Documents/PoseInsert/collect_data/run_demo.py \
        --mesh_file demo_data/peg/mesh/Peg.obj --test_scene_dir demo_data/peg_test \
        --symm_axis z --lock_symm_axis z --debug 1"
    ```

    **Track hole** — reuse rgb/depth, new mask + mesh (no re-record):
    ```bash
    # once: copy video frames, keep masks separate from peg
    mkdir -p ~/Documents/FoundationPose/demo_data/hole_test
    cp -r ~/Documents/FoundationPose/demo_data/peg_test/rgb \
          ~/Documents/FoundationPose/demo_data/peg_test/depth \
          ~/Documents/FoundationPose/demo_data/peg_test/cam_K.txt \
          ~/Documents/FoundationPose/demo_data/hole_test/

    # on host: paint hole on first frame → hole_test/masks/<first_rgb>.png
    python collect_data/paint_first_frame_mask.py \
      --scene_dir ~/Documents/FoundationPose/demo_data/hole_test

    # in docker
    docker exec -it -e DISPLAY=$DISPLAY -e XAUTHORITY=$HOME/.Xauthority -e QT_X11_NO_MITSHM=1 foundationpose bash -c \
      "python $HOME/Documents/PoseInsert/collect_data/run_demo.py \
        --mesh_file demo_data/hole/mesh/Hole.obj --test_scene_dir demo_data/hole_test \
        --symm_axis x --lock_symm_axis x --debug 1"
    ```

    **Official FP `run_demo.py`** (peg example; no symmetry / lock options):
    ```bash
    docker exec -it -e DISPLAY=$DISPLAY -e XAUTHORITY=$HOME/.Xauthority -e QT_X11_NO_MITSHM=1 foundationpose bash -c \
      "cd $HOME/Documents/FoundationPose && python run_demo.py \
        --mesh_file demo_data/peg/mesh/Peg.obj --test_scene_dir demo_data/peg_test --debug 1"
    ```

### 📷 Calibration

First of all, we use the  Cobot Mobile ALOHA, manufactured by agilex.ai.
Please calibrate the camera with the robot before data collection and evaluation to ensure correct spatial transformations between camera and the robot.

1. Place the `collect_data/ros_pose_gripper.py` to [FoundationPose](https://github.com/NVlabs/FoundationPose).
   Place the gripper in front of camera.
    ```bash
    python ros_pose_gripper.py # publish the gripper pose
    python calibation/calibation_fk.py # get the camera_in_base
    ```

### 📷 Collect

Human demonstrations is collected.

1. Place the `collect_data/ros_pose_source2.py` and `collect_data/ros_pose_target2.py`to [FoundationPose](https://github.com/NVlabs/FoundationPose).
    And get the source/target object pose.
   ```bash
    python ros_pose_source2.py # publish the source object pose
    python ros_pose_target2.py # publish the target object pose
    ```

2. Collect the train data. Remember to adjust the `save_dir`  in `collect_pose.py` .
   ```bash
    python collect_pose.py --idx 0
    ```
   
3. Show the train data.
   ```bash
    python replay_with_workspace.py 
    ```
   
4. Show the train trajectories.
   ```bash
    python vis_traj.py 
    ```
   
### 📷 Train and Test

1. Get the workspace for normalization.
   ```bash
    python get_workspace.py 
    ```

2. Train the model. Remember to adjust the ` data_path` and `ckpt_dir`.
   ```bash
    python train_pose.py 
    python train_RPDP.py 
    ```

3. Test. Remember to adjust the `ckpt` .
   ```bash
    python eval_pose.py 
    python eval_RPDP.py 
    ```
   
4. Test in the real-world. Before start the policy, the robot should grasp the source object.
   ```bash
    python device/robot_bringup.py # start up the robot
    python eval_ros_Pose.py 
    python eval_ros_RPDP.py 
    ```
   

## 📈 PoseInsert Policy

The PoseInsert policy consists of (1) a pose encoder ([`policy/cnn.py`](policy/cnn.py)), (2) a RGBD  encoder ([`policy/cnn.py`](policy/cnn.py)), (3) a Pose-Guided Residual Gated Fusion ([`policy/cnn.py`](policy/cnn.py)) and (4) a diffusion action decoder ([`policy/diffusion.py`](policy/diffusion.py)).

## 🙏 Acknowledgement

- Our diffusion module is adapted from [RISE](https://github.com/rise-policy/rise).
- Our RGBD  encoder is adapted from [FoundationPose](https://github.com/NVlabs/FoundationPose).
.


## ✍️ Citation

```bibtex
@article{sun2025exploringposeguidedimitationlearning,
    title   = {Exploring Pose-Guided Imitation Learning for Robotic Precise Insertion},
    author  = {Han Sun and Yizhao Wang and Zhenning Zhou and Shuai Wang and Haibo Yang and Jingyuan Sun and Qixin Cao},
    journal = {arXiv preprint arXiv:2404.12281},
    year    = {2025}
}
```

## 📃 License
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/sunhan1997/PoseInsert">PoseInsert</a> (including data and codebase) by 
 is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>
