import os

print(__file__)
config_file_dir = os.path.join(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1]))

DATA_DIR = os.path.join(config_file_dir, 'data')
en2fr_data = os.path.join(DATA_DIR, 'kor.txt')
