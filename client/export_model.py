import os
import version

import torch
import tensorrt as trt
from tensorrtserver.api import model_config_pb2 as model_config

from . import parse_utils
from deepclean_prod.nn.net import DeepClean
from deepclean_prod.config import config


torch.set_default_tensor_type(torch.cuda.FloatTensor)


def soft_makedirs(path):
    # basically just a reminder to myself to
    # get rid of this function and replace it
    # with the exist_ok syntax when I'm confident
    # I have the right version
    try:
        os.makedirs(path, exist_ok=True)
    except TypeError:
        if not os.path.exists(path):
            os.makedirs(path)


def convert_to_tensorrt(model_store_dir, base_config, use_fp16=False):
    trt_config = model_config.ModelConfig(
        name="deepclean_trt" + ("_fp16" if use_fp16 else "_fp32"),
        platform="tensorrt_plan",
    )
    trt_config.MergeFrom(base_config)

    trt_dir = os.path.join(model_store_dir, trt_config.name)
    soft_makedirs(os.path.join(trt_dir, "0"))

    # set up a plan builder
    TRT_LOGGER = trt.Logger()
    builder = trt.Builder(TRT_LOGGER)
    builder.max_workspace_size = 1 << 28  # 256 MiB
    builder.max_batch_size = 1  # flags['batch_size']
    if use_fp16:
        builder.fp16_mode = True
        builder.strict_type_constraints = True

    #   config = builder.create_builder_config()
    #   profile = builder.create_optimization_profile()
    #   min_shape = tuple([1] + onnx_config.input[0].dims[1:])
    #   max_shape = tuple([8] + onnx_config.input[0].dims[1:])

    #   optimal_shape = max_shape
    #   profile.set_shape('input', min_shape, optimal_shape, max_shape)
    #   config.add_optimization_profile(profile)

    # initialize a parser with a network and fill in that
    # network with the onnx file we just saved
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    onnx_path = os.path.join(model_store_dir, "deepclean_onnx", "0", "model.onnx")
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    #   last_layer = network.get_layer(network.num_layers - 1)
    #   if not last_layer.get_output(0):
    #     network.mark_output(last_layer.get_output(0))

    # build an engine from that network
    engine = builder.build_cuda_engine(network)
    with open(os.path.join(trt_dir, "0", "model.plan"), "wb") as f:
        f.write(engine.serialize())

    # export config
    with open(os.path.join(trt_dir, "config.pbtxt"), "w") as f:
        f.write(str(trt_config))


def main(flags):
    # TODO: what's the input dimensionality?
    input_dim = flags["clean_kernel"] * flags["fs"]

    # define a base config that has all the universal
    # properties. TODO: add in metadata recording
    # fs and kernel size (or really just one of them
    # since we can always do some quick math to get
    # the other from the input shape)
    base_config = model_config.ModelConfig(
        # max_batch_size=flags['batch_size'],
        input=[
            model_config.ModelInput(
                name="input",
                data_type=model_config.TYPE_FP32,
                dims=[flags["batch_size"], len(flags["chanslist"]), input_dim],
            )
        ],
        output=[
            model_config.ModelOutput(
                name="output",
                data_type=model_config.TYPE_FP32,
                dims=[flags["batch_size"], input_dim],
            )
        ],
        instance_group=[
            model_config.ModelInstanceGroup(
                count=1, kind=model_config.ModelInstanceGroup.KIND_GPU
            )
        ],
    )

    # do onnx export
    # start by copying config and setting up
    # modelstore directory
    onnx_config = model_config.ModelConfig(
        name="deepclean_onnx", platform="onnxruntime_onnx"
    )
    onnx_config.MergeFrom(base_config)

    onnx_dir = os.path.join(flags["model_store_dir"], onnx_config.name)
    soft_makedirs(os.path.join(onnx_dir, "0"))

    # create dummy input and network and use to export onnx
    dummy_input = torch.randn(*base_config.input[0].dims)
    #   dynamic_axes = {}
    #   for i in onnx_config.input:
    #     dynamic_axes[i.name] = {0: 'batch'}
    #   for i in onnx_config.output:
    #     dynamic_axes[i.name] = {0: 'batch'}

    model = DeepClean(len(flags["chanslist"]))
    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(onnx_dir, "0", "model.onnx"),
        verbose=True,
        input_names=[i.name for i in onnx_config.input],
        output_names=[i.name for i in onnx_config.output],
        #     dynamic_axes=dynamic_axes
    )

    # write config
    with open(os.path.join(onnx_dir, "config.pbtxt"), "w") as f:
        f.write(str(onnx_config))

    # do trt conversion
    convert_to_tensorrt(flags["model_store_dir"], base_config, use_fp16=False)
    convert_to_tensorrt(flags["model_store_dir"], base_config, use_fp16=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chanslist",
        type=str,
        nargs="+",
        required=True,
        action=parse_utils.ChannelListAction,
        help="Number of input channels",
    )
    parser.add_argument(
        "--clean_kernel",
        type=float,
        default=config.DEFAULT_CLEAN_KERNEL,
        help="Length of each segment in seconds",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=config.DEFAULT_SAMPLE_RATE,
        help="Sampling frequency",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size to export with"
    )
    parser.add_argument(
        "--model_store_dir",
        type=str,
        default="modelstore",
        help="Path to root model store directory",
    )

    parser = parse_utils.build_parser(parser)
    flags = parser.parse_args()
    main(vars(flags))
