import os
from threading import Thread, Event
from queue import Queue, Empty
import numpy as np
import time

from gwpy.timeseries import TimeSeriesDict

from tritongrpcclient import \
    InferenceServerClient, InferenceServerException, \
    InferInput, model_config_pb2
from tritonclientutils import triton_to_np_dtype

from deepclean_prod import config
from . import parse_utils


def get_parser():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), usage="%(prog)s [options]"
    )

    # Dataset arguments
    # TODO: can we expose channel-wise inputs on the server side and
    # do concatenation/windowing there as custom backend? Would prevent
    # unnecessary data transfer. Possibly even do post processing there
    # and just send back one sample that's at the center of the batch?
    # Will depend on extent to which network time is bottlenecking
    parser.add_argument("--clean-t0", help="GPS of the first sample", type=int)
    parser.add_argument("--clean-duration", help="Duration of frame", type=int)
    parser.add_argument(
        "--chanslist",
        help="Path to channel list",
        action=parse_utils.ChannelListAction,
        type=str
    )
    parser.add_argument(
        "--fs",
        help="Sampling frequency",
        default=config.DEFAULT_SAMPLE_RATE,
        type=float,
    )

    # Timeseries arguments
    # TODO: can we save these as metadata to the model
    # and read them as a ModelMetadataRequest? Kernel
    # size we can even infer from input shape and fs
    parser.add_argument(
        "--clean-kernel",
        help="Length of each segment in seconds",
        default=config.DEFAULT_CLEAN_KERNEL,
        type=float,
    )
    parser.add_argument(
        "--clean-stride",
        help="Stride between segments in seconds",
        default=config.DEFAULT_CLEAN_STRIDE,
        type=float,
    )
    parser.add_argument(
        "--pad-mode", help="Padding mode", default=config.DEFAULT_PAD_MODE, type=str
    )

    # Post-processing arguments
    parser.add_argument(
        "--window",
        help="Window to apply to overlap-add",
        default=config.DEFAULT_WINDOW,
        type=str,
    )

    # Input/output arguments
    parser.add_argument("--ppr-file", help="Path to preprocessing setting", type=str)
    parser.add_argument("--out-channel", help="Name of output channel", type=str)
    parser.add_argument("--log", help="Log file", type=str)

    # Server arguments
    parser.add_argument("--url", help="Server url", default="localhost:8001", type=str)
    parser.add_argument(
        "--model-name", help="Name of inference model", required=True, type=str
    )
    parser.add_argument(
        "--model-version", help="Model version to use", default=0, type=int
    )
    # TODO: can we just get this from the input_shape until we
    # get dynamic batching working?
    parser.add_argument(
        "--batch_size",
        help="Number of windows to infer on at once",
        required=True,
        type=int,
    )
    return parser


# TODO: can this just be achieved with a generator
# function passed as `target` with the args passed
# to `args`? Not clear how to kill that
# Possibly buffer class that has q object and an
# __iter__ method?
# TODO: see todo above StreamingInputTensor about
# implementing as a Process with a Pipe connection
# out from here to the StreamingInputTensor
class DataGenerator(Thread):
    def __init__(self, channels, t0, duration, fs, qsize=100):  # TODO: add as cl arg
        # If chanslist is supplied as a list, I'm assuming you've
        # already removed the target channel from it. Otherwise,
        # I'll assume the target channel is the first (0th) element
        # in a chanslist file
        if isinstance(channels, str):
            with open(channels, "r") as f:
                channels = f.read().split("\n")[1:]

        self.data = TimeSeriesDict.fetch(
            channels, t0, t0 + duration, nproc=4, allow_tape=True
        )
        self.data.resample(fs)
        self.sleep_time = 1.0 / fs
        self.channels = sorted(channels)

        self.q = Queue(maxsize=qsize)
        self._stop_event = Event()

        super(DataGenerator, self).__init__(**kwargs)

    def stop(self):
        self._stop_event.set()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        idx = 0
        while not self.stopped:
            # TODO: account for the finite time
            # the data generation takes
            time.sleep(self.sleep_time)
            samples = {}
            for channel in self.channels:
                samples[channel] = self.data[channel][idx].value

            # TODO: do some kind of timeout try-except here?
            self.q.put(samples)

            # TODO: do some sort of padding rather than
            # just doing wrapping? Would this even be
            # handled here?
            idx += 1
            idx %= len(self.data[channel])

        self.q.queue.clear()

    def get(self, timeout=None):
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            return None


# TODO: implement as separate Process with two Pipe
# connections, one coming in from data generator,
# the other going out to main process. Handle
# preproc here
class StreamingInputTensor:
    def __init__(self, chanslist, fs, kernel_length):
        if isinstance(channels, str):
            with open(channels, "r") as f:
                channels = f.read().split("\n")[1:]
        self.channels = sorted(channels)
        self.dim1 = int(kernel_length * fs)
        self._data = np.array([[0.0] for _ in channels], dtype=np.float32)

    @property
    def valid(self):
        return self._data.shape[1] == self.dim1

    def update(self, samples):
        sample = np.array([[samples[channel] for channel in self.channels]]).T
        self._data = np.concatenate([self._data, sample])
        if self._data.shape[1] > self.dim1:
            # TODO: should never be more than 1 but let's
            # be general to be safe
            overflow = self._data.shape[1] - self.dim1
            self._data = self._data[:, overflow:]

    @property
    def value(self):
        return self._data.copy()


def infer():
    pass


def post_process():
    pass


def get_datatype(model_input):
    data_type_name = model_config_pb2.DataType.Name(model.input.data_type)
    return data_type_name.split("_", maxsplit=1)[1]


def main(flags):
    # set up server connection and check that
    # server is active
    client = InferenceServerClient(flags["url"])
    if not client.is_server_live()
        raise RuntimeError("Server not live")

    # verify that model is ready
    if not client.is_model_ready(flags["model_name"]):
        try:
            client.load_model(flags["model_name"])

        # if we can't load the model, first check if it's
        # even a valid name. If it is, throw our hands up
        except InferenceServerException:
            models = client.get_model_repository_index().models
            model_names = [model.name for model in models]
            if flags["model_name"] not in model_names:
                raise ValueError(
                    "Model name {} not one of available "
                    "models: {}".format(
                        flags["model_name"], ", ".join(model_names)
                ))
            else:
                raise RuntimeError(
                    "Couldn't load model {} for unknown reason".format(
                        flags["model_name"]
                ))
    assert client.is_model_ready(flags["model_name"])
    model_metadata = client.get_model_metadata(flags["model_name"])

    # TODO: how to check version policy, or even load/unload
    # specific versions?
    assert model_metadata.versions[0] == flags["model_version"]

    # initialize an input object that we'll set to different values
    client_input = InferInput(
        model_metadata.input[0].name,
        (flags["batch_size"],) + tuple(model_metadata.input[0].dims),
        get_datatype(model_metadata.input[0])
    )

    # start data generation process
    data_generator = DataGenerator(
        flags["chanslist"], flags["clean_t0"], flags["clean_duration"], flags["fs"]
    )
    data_generator.start()

    # instantiate and initialize data in streaming
    # input tensor
    input_tensor = StreamingInputTensor(
        flags["chanslist"], flags["fs"], flags["kernel_length"]
    )
    while not input_tensor.valid:
        sample = data_generator.get()
        input_tensor.update(sample)

    with open(flags["ppr_file"], "rb") as f:
        ppr = pickle.load(f)
        mean = ppr["mean"]
        std = ppr["std"]
        # TODO: is this only for the target channel?
        filt_fl = ppr["filt_fl"]
        filt_fh = ppr["filt_fh"]
        filt_order = ppr["filt_order"]

    sample_stride = flags["clean_stride"] * flags["fs"]

    # initialize an empty numpy array that we'll fill with
    # the values from our streaming tensor and then assign
    # to our client input
    dtype = triton_to_np_dtype(client_input.datatype())
    X = np.empty(client_input.shape(), dtype=dtype)
    try:
        while True:
            # TODO: make this asynchronous in a separate
            # thread as well
            for i in range(flags["batch_size"]):
                for _ in range(sample_stride):
                    sample = data_generator.get()
                    input_tensor.update(sample)

                x = input_tensor.value
                # TODO: do we need to preprocess each separately?
                # how local are preprocessing transforms? E.g.
                # normalization is probably fine to do at sample
                # level
                x = (x - mean) / std
                # TODO: see question about target channel
                # x = bandpass(x, filt_fl, filt_fh, filt_order)
                X[i] = x.astype(dtype)

            # TODO: infer input names by doing model
            # status request
            client_input.set_data_from_numpy(X)
            triton_client.async_infer(
                model_name=flags["model_name"],
                model_version=flags["model_version"],
                inputs=[client_input],
                callback=postprocess
            )

    except Exception as e:
        data_generator.stop()
        data_generator.join()
        raise e


if __name__ == "__main__":
    parser = build_parser(get_parser())
    flags = parser.parse_args()
    main(flags)
