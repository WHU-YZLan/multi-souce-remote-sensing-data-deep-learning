from models.hrnet import seg_hrnet
from models.unet import vgg_unet, unet
# from models.hrnet_gpt import hrnet
# from keras import optimizers
from tensorflow import optimizers
import os
# 设置 TF_CPP_MIN_LOG_LEVEL 环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置 TF_FORCE_GPU_ALLOW_GROWTH 环境变量
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

lr_rate = 0.01

adam = optimizers.Adam(lr=lr_rate, decay=0.002)

model = seg_hrnet(input_height=1024, input_width=1024, channels=3, n_classes=3)

model.train(train_images="E:/0724/train/",
          train_annotations="E:/0724/train_label/",
          input_height=1024,
          input_width=1024,
          n_classes=3,
          verify_dataset=True,
          checkpoints_path="E:/0724/hrnet",
          learning_rate=lr_rate,
          epochs=200,
          batch_size=1,
          validate=True,
          val_images="E:/0724/val/",
          val_annotations="E:/0724/val_label/",
          val_batch_size=1,
          auto_resume_checkpoint=True,
          load_weights=None,
          steps_per_epoch=1820,
          val_steps_per_epoch=280,
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name=adam,
          do_augment=False,
          augmentation_name="aug_all",
          callbacks=None,
          custom_augmentation=None,
          other_inputs_paths=None,
          preprocessing=None,
          read_image_type=1  # cv2.IMREAD_COLOR = 1 (rgb),
                             # cv2.IMREAD_GRAYSCALE = 0,
                             # cv2.IMREAD_UNCHANGED = -1 (4 channels like RGBA)
         )