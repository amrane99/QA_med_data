# ------------------------------------------------------------------------------
# Module where paths should be defined.
# ------------------------------------------------------------------------------
import os

# Path where intermediate and final results are stored
storage_path = '/local/scratch/Racoon_tn/Thorax_QA'
storage_data_path = os.path.join(storage_path, 'data')

# Original data paths. TODO: set necessary data paths.
original_data_paths = {'batch1': '/local/scratch/Racoon_tn/Thorax_QA'}

# Path that represents JIP data structure for training and inference
JIP_dir = os.path.join(storage_path, 'JIP')

# Login for Telegram Bot
telegram_login = {'chat_id': 'TODO', 'token': 'TODO'}