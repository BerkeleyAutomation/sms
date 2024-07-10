"""
sms configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.trainer import TrainerConfig as TrainerConfigBase
from sms.encoders.openclip_encoder import OpenCLIPNetworkConfig
# from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig
from sms.model.sms_gaussian_splatting import smsGaussianSplattingModelConfig
# from sms.monodepth.zoedepth_network import ZoeDepthNetworkConfig
# from sms.monodepth.midas_network import MidasDepthNetworkConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

# from sms.sms_trainer import TrainerConfig
# from sms.sms_pipeline import smsPipelineConfig
from sms.sms_data_pipeline import smsdataPipelineConfig
# from sms.data.sms_datamanager import smsDataManagerConfig, smsDataManager
from sms.data.full_images_datamanager import FullImageDatamanagerConfig
# from sms.data.sms_dataparser import smsDataParserConfig
# from sms.data.sms_dataset import smsDataset


# sms_method = MethodSpecification(
#     config = TrainerConfig(
#         method_name="sms",
#         steps_per_eval_image=100,
#         steps_per_eval_batch=100,
#         steps_per_save=2000,
#         steps_per_eval_all_images=100000, 
#         max_num_iterations=30000,
#         mixed_precision=False,
#         gradient_accumulation_steps = {'camera_opt': 100,'color':10,'shs':10},
#         pipeline=smsPipelineConfig(
#             datamanager=smsDataManagerConfig(
#                 _target=smsDataManager[smsDataset],
#                 dataparser=smsDataParserConfig(),
                
#                 #  You can swap the type of input encoder by specifying different NetworkConfigs, the one below uses OpenAI CLIP, the one above uses OpenCLIP
#                 # image_encoder=CLIPNetworkConfig(
#                 #     clip_model_type="ViT-B/16", clip_n_dims=512
#                 # )
#             ),
#             model=smsGaussianSplattingModelConfig(),
#             network=OpenCLIPNetworkConfig(
#                     clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512, device='cuda:0'
#             ),
#             # depthmodel=ZoeDepthNetworkConfig(device='cuda:0'),
#         ),
#         optimizers={
#             "means": {
#                 "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(
#                     lr_final=1.6e-6,
#                     max_steps=30000,
#                 ),
#             },
#             "features_dc": {
#                 "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
#                 "scheduler": None,
#             },
#             "features_rest": {
#                 "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
#                 "scheduler": None,
#             },
#             "opacities": {
#                 "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
#                 "scheduler": None,
#             },
#             "scales": {
#                 "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000)
#             },
#             "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
#             "camera_opt": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
#             },
#              "lerf": {
#                 "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=15000),
#             },
#             "appearance_embed": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": None,
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Base config for Lifelong Language Embedded Gaussian Splatting",
# )


sms_data_method = MethodSpecification(
    config = TrainerConfigBase(
        method_name="sms-data",
        steps_per_eval_image=100,
        steps_per_eval_batch=100,
        steps_per_save=2000,
        steps_per_eval_all_images=100000, 
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps = {'camera_opt': 100,'color':10,'shs':10},
        pipeline=smsdataPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                network=OpenCLIPNetworkConfig(
                    clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512, device='cuda:0'
                ),
                # _target=smsDataManager[smsDataset],
                # dataparser=smsDataParserConfig(),
                
                #  You can swap the type of input encoder by specifying different NetworkConfigs, the one below uses OpenAI CLIP, the one above uses OpenCLIP
                # image_encoder=CLIPNetworkConfig(
                #     clip_model_type="ViT-B/16", clip_n_dims=512
                # )
            ),
            model=smsGaussianSplattingModelConfig(),

            # network=OpenCLIPNetworkConfig(
            #         clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512, device='cuda:0'
            #     ),
            
            # depthmodel=ZoeDepthNetworkConfig(device='cuda:0'),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000)
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
             "lerf": {
                "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=15000),
            },
            "dino_feats": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=6000,
                ),
            },
            "nn_projection": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=6000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for Lifelong Language Embedded Gaussian Splatting",
)
