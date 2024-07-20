import os
import logging
from tensorboardX import SummaryWriter
import numpy as np

class MyFormatter(logging.Formatter):

    def __init__(self):
        super().__init__()
        self.formats = {
            logging.DEBUG: logging.Formatter('%(message)s'),
            logging.INFO: logging.Formatter('%(asctime)s - INFO - %(message)s'),
            logging.WARNING: logging.Formatter('%(asctime)s - WARNING - %(message)s'),
            logging.ERROR: logging.Formatter('%(asctime)s - ERROR - %(message)s'),
            logging.CRITICAL: logging.Formatter('%(asctime)s - CRITICAL - %(message)s')
        }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        return log_fmt.format(record)
    
def get_txt_logger(log_dir):
    logging.basicConfig(level=logging.DEBUG, filemode='w')
    logger = logging.getLogger(__name__)
    logger.removeHandler(logging.StreamHandler())
    
    handler = logging.FileHandler(log_dir)
    handler.setLevel(logging.DEBUG)
    formatter = MyFormatter()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger

class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        txt_dir = os.path.join(log_dir, "log.log")
        self.text_logger = get_txt_logger(txt_dir)
        self.text_logger.debug('#'*80)
        self.text_logger.debug('logging outputs to {}'.format(log_dir))
        self.text_logger.debug('#'*80)
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)  # [C, H, W]
        self._summ_writer.add_image('{}'.format(name), image, step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):

        # reshape the rollouts
        videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in paths]

        # max rollout length
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0]>max_length:
                max_length = videos[i].shape[0]

        # pad rollouts to all be same length
        for i in range(max_videos_to_save):
            if videos[i].shape[0]<max_length:
                padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        # log videos to tensorboard event file
        videos = np.stack(videos[:max_videos_to_save], 0)
        self.log_video(videos, video_title, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)
        
    def plot_graph(array):
        pass

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def flush(self):
        self._summ_writer.flush()