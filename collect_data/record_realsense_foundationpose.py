## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.
## Adapted from https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043

#####################################################
##  Record aligned RealSense RGB-D for FP run_demo ##
#####################################################

import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record aligned RealSense RGB-D in FoundationPose run_demo format."
    )
    default_out = os.path.expanduser("~/Documents/FoundationPose/demo_data/peg_test")
    parser.add_argument(
        "-o",
        "--output",
        default=default_out,
        help=f"Output scene dir (rgb/, depth/, cam_K.txt). Default: {default_out}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output)
    subfolder_depth = os.path.join(output_dir, "depth")
    subfolder_rgb = os.path.join(output_dir, "rgb")
    subfolder_depth_unaligned = os.path.join(output_dir, "depth_unaligned")
    subfolder_rgb_unaligned = os.path.join(output_dir, "rgb_unaligned")

    for path in (
        subfolder_depth,
        subfolder_rgb,
        subfolder_depth_unaligned,
        subfolder_rgb_unaligned,
    ):
        os.makedirs(path, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for sensor in device.sensors:
        if sensor.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        raise RuntimeError("This script requires a RealSense with an RGB sensor.")

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if device_product_line == "L500":
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth scale:", depth_scale)
    print("Output dir:", output_dir)

    clipping_distance_in_meters = 1.0
    clipping_distance = clipping_distance_in_meters / depth_scale
    align = rs.align(rs.stream.color)

    record_stream = False
    cam_k_path = os.path.join(output_dir, "cam_K.txt")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            unaligned_depth_frame = frames.get_depth_frame()
            unaligned_color_frame = frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            unaligned_depth_image = np.asanyarray(unaligned_depth_frame.get_data())
            unaligned_rgb_image = np.asanyarray(unaligned_color_frame.get_data())

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
            cv2.imshow("Align Example", images)

            key = cv2.waitKey(1)

            if key & 0xFF == ord(" "):
                if not record_stream:
                    time.sleep(0.2)
                    record_stream = True
                    with open(cam_k_path, "w", encoding="utf-8") as file:
                        file.write(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
                        file.write(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
                        file.write(f"{0.0} {0.0} {1.0}\n")
                    print("Recording started")
                else:
                    record_stream = False
                    print("Recording stopped")

            if record_stream:
                framename = int(round(time.time() * 1000))
                cv2.imwrite(os.path.join(subfolder_depth, f"{framename}.png"), depth_image)
                cv2.imwrite(os.path.join(subfolder_rgb, f"{framename}.png"), color_image)
                cv2.imwrite(
                    os.path.join(subfolder_depth_unaligned, f"{framename}.png"),
                    unaligned_depth_image,
                )
                cv2.imwrite(
                    os.path.join(subfolder_rgb_unaligned, f"{framename}.png"),
                    unaligned_rgb_image,
                )

            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
