import tkinter as tk
from typing import Literal
from PIL import Image, ImageTk
import subprocess
import numpy as np
import threading
from using import use
from PyXiNet import PyXiNetA1
from Config import Config
import torch
from matplotlib import cm


class Webcam:
    def __init__(
        self, root: tk.Tk, env: Literal["HomeLab", "Cluster"], model: torch.nn.Module
    ):
        """
        # Warning
        This class has only been tested on macOS, with a Macbook M1 pro chip.
        """
        self.root = root
        self.root.title("Video della fotocamera")

        # Configura il display della fotocamera
        self.label = tk.Label(root)
        self.label.pack()

        # Buffer per il frame corrente
        self.current_frame = None
        self.lock = threading.Lock()

        # Avvia la fotocamera con ffmpeg
        self.command = [
            "ffmpeg",
            "-f",
            "avfoundation",
            "-framerate",
            "30",
            "-video_size",
            "640x480",
            "-i",
            "0",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-an",
            "-sn",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self.proc = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, bufsize=10**8
        )

        # Thread per leggere i frame
        self.read_thread = threading.Thread(target=self.read_frames)
        self.read_thread.daemon = True
        self.read_thread.start()

        # Model usage
        self.config = Config(env).get_configuration()
        if (
            self.config.checkpoint_to_use_path == None
            or self.config.checkpoint_to_use_path == ""
        ):
            print("You have to select a checkpoint to correctly configure the model.")
            exit(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model creation and configuration
        self.model = model.to(self.device)
        checkpoint = torch.load(
            self.config.checkpoint_to_use_path,
            map_location=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Aggiorna il frame della fotocamera
        self.update_frame()

    def read_frames(self):
        while True:
            raw_frame = self.proc.stdout.read(640 * 480 * 3)
            if len(raw_frame) != 640 * 480 * 3:
                break
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((480, 640, 3))

            with self.lock:
                self.current_frame = frame

    def update_frame(self):
        with self.lock:
            frame = self.current_frame

        if frame is not None:
            image = Image.fromarray(frame)

            image: np.ndarray = (
                use(
                    self.model,
                    image,
                    self.config.image_width,
                    self.config.image_height,
                    image.size[0],
                    image.size[1],
                    self.device,
                )
                .cpu()
                .numpy()
            )
            disparity_normalized = (image - np.min(image)) / (
                np.max(image) - np.min(image)
            )
            disparity_colored = cm.plasma(
                disparity_normalized
            )  # 'plasma' è solo un esempio, puoi scegliere qualsiasi mappa di colori disponibile

            # Converti da colore mappato (RGBA) a formato immagine PIL accettabile da Tkinter
            disparity_colored = (disparity_colored[:, :, :3] * 255).astype(
                np.uint8
            )  # Prendi solo i canali RGB, ignora alpha
            pil_image = Image.fromarray(disparity_colored)

            image_tk = ImageTk.PhotoImage(pil_image)

            # Aggiorna l'immagine sulla label
            self.label.config(image=image_tk)
            self.label.image = image_tk

        # Richiama la funzione per aggiornare il frame
        self.root.after(10, self.update_frame)

    def __del__(self):
        # Termina il processo ffmpeg
        self.proc.terminate()
