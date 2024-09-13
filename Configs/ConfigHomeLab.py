class ConfigHomeLab:

    def __init__(self):
        self.model_name: str = "PyXiNet"
        self.data_path: str = "/media/riccardo-toniolo/Volume/KITTI/"
        self.filenames_file_training: str = "./filenames/eigen_train_files_png.txt"
        self.filenames_file_testing: str = "./filenames/eigen_test_files_png.txt"
        self.batch_size: int = 8
        self.num_epochs: int = 50
        self.learning_rate: float = 1e-4
        self.image_width: int = 512
        self.image_height: int = 256
        self.shuffle_batch: bool = True

        self.alpha = 0.4
        self.gamma = 4

        self.weight_lr: float = 1
        # Left-right consistency weight in the total loss calculation

        self.weight_ap: float = 1
        # Reconstruction error loss weight in the total loss calculation

        self.weight_df: float = 0.1
        # Disparity gradient weight in the total loss calculation

        self.weight_SSIM: float = 0.85
        # Weight between SSIM and L1 in the image loss

        self.output_directory: str = "./outputfiles/outputs/"
        # Output directory for the disparities file and for cluster
        # logs (if you use slurm files)
        self.checkpoint_path: str = "./outputfiles/checkpoints/"
        # Directory to be used to store checkpoint files.

        self.retrain: bool = True
        # If True it retrains the model without using checkpoints

        self.debug: bool = True
        # Not used anymore but useful to enable certain code sections only when this
        # parameter is set to True.

        self.checkpoint_to_use_path: str = (
            "./outputfiles/checkpoints/betaCBAM1/betaCBAM1_e048.pth.tar"
        )
        # Path of the checkpoint file to be used inside the model.

        self.disparities_to_use: str = "./outputfiles/outputs/disparities.npy"
        # Path of the disparities file to be used for evaluations.

        self.test_dir_for_inference_time = "../../10_test_images/"
        # Path of the directory used to measure the avg inference time with CPU on 10 images.

        ########################## EXPERIMENTS PARAMETERS ##########################
        # This parameters are only used to test the behaviour of the model and the #
        # training using specific conditions. All of them are meant to be turned   #
        # off for PyDNet to be trained as the original paper meant.                #
        ############################################################################

        self.HSV_processing: bool = False
        # It means that images will be processed in HSV format instead of RGB

        self.BlackAndWhite_processing: bool = False
        # It means that images will be processed only in the gray scale (single channel)

        self.VerticalFlipAugmentation: bool = False
        # It means that images will have a 50% chance of being flipped upside-down

        self.KittiRatioImageSize: bool = False
        # It will use a 640x192 size for input images.

        self.RatioImageSize1024x320: bool = False
        # It will use a 1024x320 size for input images.

        self.RatioImageSize1280x384: bool = False
        # It will use a 1280x384 size for input images.

        self.PyDNet2_usage: bool = False
        # It means that the model that will be used is PyDNet2 instead of PyDNet
        count = 0
        if self.HSV_processing:
            self.checkpoint_path += "HSV/"
            count += 1
        elif self.BlackAndWhite_processing:
            self.checkpoint_path += "BandW/"
            count += 1
        elif self.KittiRatioImageSize:
            self.checkpoint_path += "640x192/"
            self.image_height = 192
            self.image_width = 640
            count += 1
        elif self.RatioImageSize1024x320:
            self.checkpoint_path += "1024x320/"
            self.image_height = 320
            self.image_width = 1024
            count += 1
        elif self.RatioImageSize1280x384:
            self.checkpoint_path += "1280x384/"
            self.image_height = 384
            self.image_width = 1280
            count += 1
        elif self.PyDNet2_usage:
            self.checkpoint_path += "PyDNet2/"
            count += 1
        elif self.VerticalFlipAugmentation:
            self.checkpoint_path += "VFlip/"
            count += 1
        if self.PyDNet2_usage:
            self.model_name = self.model_name.replace("V1", "V2")
        if count > 1:
            raise Exception(
                "Can't have more than one experimental configuration turned on!"
            )
