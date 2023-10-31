import os

def list_current_directory():
    # Get the current working directory, expecting /home/ActionFeature in docker file system
    current_dir = os.getcwd()
    print(f'Current Directory: {current_dir}')

    # List files and directories in the current working directory
    print('Contents:')
    for item in os.listdir(current_dir):
        print(item)

