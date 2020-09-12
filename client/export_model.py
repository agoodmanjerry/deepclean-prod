import contextlib
import logging
import os
import shutil

import torch
from tritongrpcclient import model_config_pb2 as model_config

import parse_utils
from deepclean_prod.nn.net import DeepClean
from deepclean_prod import config


torch.set_default_tensor_type(torch.cuda.FloatTensor)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def soft_makedirs(path):
    # basically just a reminder to myself to
    # get rid of this function and replace it
    # with the exist_ok syntax when I'm confident
    # I have the right version (os doesn't have
    # a __version__ attribute unfortunately)
    try:
        os.makedirs(path, exist_ok=True)
    except TypeError:
        if not os.path.exists(path):
            os.makedirs(path)


def convert_to_tensorrt(model_store_dir, base_config, use_fp16=False):
    logger.info("Building TensorRT model with precison {}".format(
        "fp16" if use_fp16 else "fp32")
    )
    trt_config = model_config.ModelConfig(
        name="deepclean_trt" + ("_fp16" if use_fp16 else "_fp32"),
        platform="tensorrt_plan",
    )
    trt_config.MergeFrom(base_config)

    trt_dir = os.path.join(model_store_dir, trt_config.name)
    soft_makedirs(os.path.join(trt_dir, "0"))

    # set up a plan builder
    TRT_LOGGER = trt.Logger()
    with contextlib.ExitStack() as stack:
        builder = stack.enter_context(trt.Builder(TRT_LOGGER))
        builder.max_workspace_size = 1 << 28 # 256 MiB
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
        network = stack.enter_context(builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ))

        parser = stack.enter_context(trt.OnnxParser(network, TRT_LOGGER))
        onnx_path = os.path.join(model_store_dir, "deepclean_onnx", "0", "model.onnx")
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        last_layer = network.get_layer(network.num_layers - 1)
        if not last_layer.get_output(0):
            logger.info("Marking output layer")
            network.mark_output(last_layer.get_output(0))

        # build an engine from that network
        engine = builder.build_cuda_engine(network)
        with open(os.path.join(trt_dir, "0", "model.plan"), "wb") as f:
            f.write(engine.serialize())

        # export config
        with open(os.path.join(trt_dir, "config.pbtxt"), "w") as f:
            f.write(str(trt_config))


def main(flags):
    input_dim = int(flags["clean_kernel"] * flags["fs"])

    # define a base config that has all the universal
    # properties. TODO: add in metadata recording
    # fs and kernel size (or really just one of them
    # since we can always do some quick math to get
    # the other from the input shape)
    flags["chanslist"] = flags["chanslist"][1:]
    logger.info("Building model from channels:{}".format(
        "\n\t".join(flags["chanslist"])
    ))
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

    # TODO: add in command line args for selecting which
    # type of export to do, including an "all" option
    model = DeepClean(len(flags["chanslist"]))
    print(model(dummy_input).shape)

    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(onnx_dir, "0", "model.onnx"),
        verbose=True,
        input_names=[i.name for i in onnx_config.input],
        output_names=[i.name for i in onnx_config.output],
        #     dynamic_axes=dynamic_axes
    )

    # do trt conversion
    if "trt-fp32" in flags["export_as"]:
        convert_to_tensorrt(flags["model_store_dir"], base_config, use_fp16=False)
    if "trt-fp16" in flags["export_as"]:
        convert_to_tensorrt(flags["model_store_dir"], base_config, use_fp16=True)

    if "onnx" in flags["export_as"]:
        with open(os.path.join(onnx_dir, "config.pbtxt"), "w") as f:
            f.write(str(onnx_config))
    else:
        logger.info("Removing onnx model")
        shutil.rmtree(onnx_dir)


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
        "--clean-kernel",
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
        "--batch-size",
        type=int,
        default=8,
        help="Batch size to export with"
    )
    parser.add_argument(
        "--model-store-dir",
        type=str,
        default="modelstore",
        help="Path to root model store directory",
    )
    parser.add_argument(
        "--export-as",
        type=str,
        nargs="+",
        choices=("onnx", "trt-fp32", "trt-fp16", "all"),
        default=["all"],
        help="Which type of format to export as"
    )

    parser = parse_utils.build_parser(parser)
    flags = parser.parse_args()
    flags = vars(flags)

    if "all" in flags["export_as"]:
        flags["export_as"] = ["onnx", "trt-fp32", "trt-fp16"]
    if any([i.startswith("trt") for i in flags["export_as"]]):
        try:
            import tensorrt as trt
        except ModuleNotFoundError as e:
            msg = (
                "Tried to export model as version(s) {}, but "
                "TensorRT library is not installed!".format(
                    ", ".join(
                        [i for i in flags["export_as"] if i.startswith("trt")])
            ))
            raise ModuleNotFoundError(msg) from e

    main(flags)
