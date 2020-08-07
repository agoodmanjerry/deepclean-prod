import os
from threading import Thread, Event
from queue import Queue, Empty
import numpy as np
import time

from gwpy.timeseries import TimeSeriesDict

import grpc
from tritongrpcclient import grpc_service_pb2
from tritongrpcclient import grpc_service_pb2_grpc

from deepclean_prod import config
from parse_utils import build_parser


def get_parser():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), usage="%(prog)s [options]"
    )

    # Dataset arguments
    # TODO: can we expose channel-wise inputs on the server side and
    # do concatenation/windowing there as custom backend?
    parser.add_argument("--clean-t0", help="GPS of the first sample", type=int)
    parser.add_argument("--clean-duration", help="Duration of frame", type=int)
    parser.add_argument("--chanslist", help="Path to channel list", type=str)
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
            # just doing this wrap? Would this even be
            # handled here?
            idx += 1
            idx %= len(self.data[channel])

        self.q.queue.clear()

    def get(self, timeout=None):
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            return None


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


def main(flags):
    # set up server connection and check that
    # server is active
    channel = grpc.insecure_channel(flags["url"])
    grpc_stub = grpc_service_pb2_grpc.GRPCInferenceServiceStub(channel)
    try:
        request = grpc_service_pb2.ServerLiveRequest()
        response = grpc_stub.ServerLive(request)
    except Exception as e:
        raise RuntimeError("Server not live") from e

    # verify that model is ready
    request = grpc_service_pb2.ModelReadyRequest(
        name=flags["model_name"], version=flags["model_version"]
    )
    response = grpc_stub.ModelReady(request)
    # TODO: some check on response

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
    times = [time.time()]
    X = np.empty((flags["batch_size"],) + input_tensor._data.shape, dtype=np.float32)
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
                X[i] = x

            inference_request = grpc_service_pb2.ModelInferRequest(
                model_name=flags["model_name"],
                model_version=flags["model_version"],
                id="deepclean",
            )
            input = grpc_service_pb2.ModelInferRequest().InferInputTensor()
            infer_ctx.run(X)
            times.append(time.time())
    except Exception as e:
        data_generator.stop()
        data_generator.join()
        raise e


if __name__ == "__main__":
    parser = build_parser(get_parser())
    flags = parser.parse_args()
    main(flags)
