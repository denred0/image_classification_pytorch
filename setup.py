from distutils.core import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
  name = 'image_classification_pytorch',         # How you named your package folder (MyLib)
  packages = ['image_classification_pytorch'],   # Chose the same as "name"
  version = '0.0.9',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Image classification with pretrained models in Pytorch',   # Give a short description about your library
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  author = 'Denis Potapov',                   # Type in your name
  author_email = 'potapovdenisdmit@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/denred0/image_classification_pytorch',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/denred0/image_classification_pytorch/archive/refs/tags/0.0.9.tar.gz',    # I explain this later on
  keywords = ['pytorch', 'image classification', 'imagenet', 'pretrained model'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pytorch_lightning',
          'torch',
          'sklearn',
          'albumentations==0.5.1',
          'opencv-python',
          'torchmetrics',
          'timm',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)