from federatedscope.register import register_model
from AttentiveNAS.models.model_factory import create_model

from federatedscope.core.configs.config import CN


# Build you torch or tf model class here
# class MyNet(object):
#     pass


# Instantiate your model class with config and data
# def ModelBuilder(model_config, local_data):
#     pass


def call_attentive_net(model_config, input_shape):
    if "attentive" in model_config.type:
        attentive_yaml = "AttentiveNAS/configs/attentive_supernet_cifar.yml"

        from AttentiveNAS.utils.config import setup
        cfg = setup(attentive_yaml)
        cfg.n_classes = model_config.get("n_classes", 10)
        cfg.bn_momentum = model_config.get("bn_momentum", 0.1)
        cfg.bn_eps = model_config.get("bn_eps", 1e-5)
        cfg.drop_out = model_config.get("drop_out", 0)
        cfg.drop_connect = model_config.get("drop_connect", 0)

        # model_cfg = CN()
        # model_cfg.set_new_allowed(True)  # TODO(Variant): to receive new config dict
        # model_cfg.merge_from_file(attentive_yaml)
        # model_cfg.bn_momentum = 0
        # model_cfg.bn_eps = 1e-5

        # NOTE: cfg.drop_out, cfg.drop_connect 无效。对于dynamic model，其drop率随着训练采样动态变化。
        if model_config.type == "attentive_supernet":
            model = create_model(cfg, "attentive_nas_dynamic_model")
            return model
        elif model_config.type == "attentive_min_subnet":
            model = create_model(cfg, "attentive_nas_dynamic_model")
            model.sample_min_subnet()
            model = model.get_active_subnet(preserve_weight=True)
            return model


register_model("attentive_net", call_attentive_net)
