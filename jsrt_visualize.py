import improc

from omegaconf import OmegaConf

p = OmegaConf.create({'data_path': '/media/gerda/WD4/Datasets',
                      'batch_size': 3,
                      'num_classes': 6})

IDs = ['JPCLN089', 'JPCLN033', 'JPCLN135']

batch_x, batch_y = improc.load_batch_JSRT(p, IDs, [0, 1, 2])

improc.plot_batch_sample(p, batch_x, batch_y, 'jsrt_sample.png')