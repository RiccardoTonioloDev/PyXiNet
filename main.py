import argparse
from training import train
from testing import generate_test_disparities
from evaluating import eval_disparities_file
from using import use_with_path, inference_time_avg_10
from webcam import Webcam
import tkinter as tk
from PyXiNet import (
    PyXiNetA1,
    PyXiNetA2,
    PyXiNetB1,
    PyXiNetB2,
    PyXiNetB3,
    PyXiNetB4,
    PyXiNetM1,
    PyXiNetM2,
    PyXiNetM3,
    PyXiNetM4,
    PyXiNetBCBAM1,
    PyXiNetBCBAM2,
    PydnetC1,
)
from Config import Config

parser = argparse.ArgumentParser(description="PyDNet pytorch implementation.")

parser.add_argument(
    "--mode",
    type=str,
    help="[train,test,eval,use]\ntrain: trains the model;\ntest: generates a disparities.npy file to be used for evaluation;\neval: evaluates the disparities.npy file;\nuse: to use it do generate a disparity heatmap from an image.",
    default="test",
)
parser.add_argument("--env", type=str, help="[HomeLab,Cluster]", default="HomeLab")
parser.add_argument(
    "--img_path",
    type=str,
    help="The path of the image to be used to make its disparity heatmap.",
)

args = parser.parse_args()

if args.env not in ["HomeLab", "Cluster"]:
    print(
        "You inserted the wrong mode argument in env. Choose between: 'HomeLab' and 'Cluster'."
    )
    exit(0)

config = Config(args.env).get_configuration()
# model = PyXiNetA1(config)
# model = PyXiNetA2(config)
# model = PyXiNetB1(config)
# model = PyXiNetB2(config)
# model = PyXiNetB3(config)
# model = PyXiNetB4(config)
# model = PyXiNetM1(config)
# model = PyXiNetM2(config)
# model = PyXiNetM3(config)
# model = PyXiNetM4(config)
model = PyXiNetBCBAM1(config)
# model = PyXiNetBCBAM2(config)
# model = PydnetC1()

if args.mode == "train":
    train(args.env, model)
elif args.mode == "test":
    generate_test_disparities(args.env, model)
elif args.mode == "eval":
    eval_disparities_file(args.env)
    inference_time_avg_10(
        config.test_dir_for_inference_time,
        model,
        config.image_height,
        config.image_width,
    )
elif args.mode == "use":
    use_with_path(args.env, args.img_path, model)
elif args.mode == "webcam":
    root = tk.Tk()
    app = Webcam(root, args.env, model)
    root.mainloop()
else:
    print(
        "You inserted the wrong mode argument in mode. Choose between: 'train', 'test', 'eval' and 'use'."
    )
