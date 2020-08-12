import os
import multiprocessing as mp
import numpy as np
import time
from abc import abstractmethod

from gwpy.timeseries import TimeSeriesDict

import tritongrpcclient as triton
from tritongrpcclient import model_config_pb2 as model_config
from tritonclientutils import triton_to_np_dtype

from deepclean_prod import config
from deepclean_prod.signal import bandpass
from . import parse_utils


# TODO: move somewhere else
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
    parser.add_argument(
        "--clean-t0",
        help="GPS of the first sample",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--clean-duration",
        help="Duration of frame",
        type=int,
        required=True
    )
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
        "--pad-mode",
        help="Padding mode",
        default=config.DEFAULT_PAD_MODE,
        type=str
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


class StoppableIteratingBuffer:
    def __init__(self, conn_in=None, conn_out=None):
        self.conn_in = conn_in
        self.conn_out = conn_out
        self._stop_event = mp.Event()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def stop(self):
        self._stop_event.set()

    def __call__(self):
        self.initialize_loop()
        while not self.stopped:
            self.loop()

        # do some general cleanup
        if self.conn_in is not None:
            self.conn_in.close()
        if self.conn_out is not None:
            self.conn_out.close()

        # allow for custom subclass cleanup as well
        self.cleanup()

    def initialize_loop(self):
        pass

    @abstractmethod
    def loop(self):
        '''
        required to have this method for main funcionality
        '''
        pass

    def cleanup(self):
        pass


class DataGeneratorBuffer(StoppableIteratingBuffer):
    # TODO: better to use queue for max size?
    def __init__(self, channels, t0, duration, fs, **kwargs):
        self.data = TimeSeriesDict.fetch(
            channels, t0, t0 + duration, nproc=4, allow_tape=True
        )
        self.data.resample(fs)

        # TODO: do something to account for finite generation
        # time?
        self.sleep_time = 1. / fs
        self.channels = sorted(channels)
        super().__init__(**kwargs)

    def initialize_loop(self):
        self.idx = 0

    def loop(self):
        time.sleep(self.sleep_time)
        samples = {}
        for channel in self.channels:
            samples[channel] = self.data[channel][idx].value

        # TODO: do some kind of timeout try-except here?
        self.conn_out.send(samples)

        # TODO: do some sort of padding rather than
        # just doing wrapping? Would this even be
        # handled here?
        self.idx += 1
        self.idx %= len(self.data[channel])


class InputDataBuffer(StoppableIteratingBuffer):
    def __init__(self,
            batch_size,
            channels,
            kernel_size,
            kernel_stride,
            fs,
            ppr_file=None,
            **kwargs
    ):
        # total number of samples in a single batch
        self.num_samples = (kernel_stride*(batch_size-1) + kernel_size)*fs

        # tells us how to window a 2D stream of data into a 3D batch
        slices = []
        for i in range(batch_size):
            start = i*kernel_stride*fs
            stop = start + kernel_size*fs
            slices.append(slice(start, stop))
        self.slices = slices

        self.channels = sorted(channels)

        # load preprocessing info if there is any
        # since we only do bandpass on the target, I'll
        # ignore those params for now
        if ppr_file is not None:
            with open(ppr_file, "rb") as f:
                ppr = pickle.load(f)
                self.mean = ppr["mean"]
                self.std = ppr["std"]
        else:
            self.mean, self.std = None, None

        super().__init__(**kwargs)

    def read_sensor(self):
        '''
        read individual samples and return an array of size
        `(len(self.channels), 1)` for hstacking
        '''
        samples = self.conn_in.recv()
        return np.array(
            [[samples[channel]] for channel in self.channels],
            dtype=np.float32 # TODO: use dtype from model metadata
        )

    def preprocess(self, data):
        '''
        perform any preprocessing transformations on the data
        just does normalization for now
        '''
        # TODO: is there any extra preprocessing that should
        # be done? With small enough strides and batch sizes,
        # does there reach a point at which it makes sense
        # to do preproc on individual samples (assuming it
        # can be done that locally) to avoid doing thousands
        # of times on the same sample? Where is this limit?
        if self.mean is not None:
            return (data - self.mean) / self.std
        else:
            return data

    def batch(self, data):
        '''
        take windows of data at strided intervals and stack them
        '''
        return np.hstack([data[:, slc] for slc in self.slices])

    def loop(self):
        # start by reading the next batch of samples
        # TODO: play with numpy to see what's most efficient
        # concat and reshape? read_sensor()[:, None]?
        data = []
        for i in range(self.num_samples):
            data.append(read_sensor())
        data = np.hstack(data)
    
        data = self.preprocess(data)
        data = self.batch(data)
        self.conn_out.send(data)

    def cleanup(self):
        self.conn_in.close()
        self.conn_out.close()


class AsyncInferenceClient(StoppableIteratingBuffer):
    def __init__(self, url, model_name, model_version, **kwargs):
        # set up server connection and check that server is active
        client = triton.InferenceServerClient(url)
        if not client.is_server_live():
            raise RuntimeError("Server not live")

        # verify that model is ready
        if not client.is_model_ready(model_name):
            # if not, try to load use model control API
            try:
                client.load_model(model_name)

            # if we can't load the model, first check if the given
            # name is even valid. If it is, throw our hands up
            except triton.InferenceServerException:
                models = client.get_model_repository_index().models
                model_names = [model.name for model in models]
                if model_name not in model_names:
                    raise ValueError(
                        "Model name {} not one of available models: {}".format(
                            model_name, ", ".join(model_names))
                    )
                else:
                    raise RuntimeError(
                        "Couldn't load model {} for unknown reason".format(
                            model_name)
                    )
            # double check that load worked
            assert client.is_model_ready(model_name)

        metadata = client.get_model_metadata(model_name)
        # TODO: find better way to check version, or even to
        # load specific version
        assert model_metadata.versions[0] == model_version

        model_input = model_metadata.input[0]
        data_type_name = model_config.DataType.Name(model_input.data_type)
        data_type = data_type_name.split("_", maxsplit=1)[1]

        model_output = model_metadata.ouptut[0]

        self.client_input = triton.InferInput(
            model_input.name, tuple(model_input.dims), data_type
        )
        self.client_output = triton.InferRequestedOutput(model_output.name)
        self.client = client
        super().__init__(**kwargs)

    def loop(self):
        X = self.conn_in.recv()
        self.client_input.set_data_from_numpy(X)
        self.client.async_infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[self.client_input],
            outputs=[self.client_output]
            callback=self.process_result
        )
    
    def process_result(self, result, error):
        # TODO: add error checking
        data = result.as_numpy(self.client_output.name())
        self.conn_out.send(data)


class PostProcessBuffer(StoppableIteratingBuffer):
    def __init__(self, ppr_file, **kwargs):
        # TODO: should we do a check on the mean and std like we
        # do during preprocessing?
        with open(ppr_file, "rb") as f:
            ppr = pickle.load(f)
            self.mean = ppr["mean"]
            self.std = ppr["std"]
            self.filt_fl = ppr['filt_fl']
            self.filt_fh = ppr['filt_fh']
            self.filt_order = ppr['filt_order']

        super().__init__(**kwargs)

    def loop(self):
        data = self.conn_in.recv()
        data = self.std*data + self.mean

        # TODO: add overlap calculation, maybe keep track of streaming
        # output tensor to capture overlap from predictions that
        # weren't in this batch?
        data = bandpass(data, self.filt_fl, self.filt_fh, self.filt_order)

        # TODO: keep track of running throughput average, find a way
        # to capture latency (maybe by reference to data_generator.idx)?
        # stream throughput data somewhere


def main(flags):
    raw_data_in, raw_data_out = mp.Pipe(duplex=False)
    preproc_data_in, preproc_data_out = mp.Pipe(duplex=False)
    infer_result_in, infer_result_out = mp.Pipe(duplex=False)

    raw_data_buffer = DataGeneratorBuffer(
        flags["chanslist"],
        flags["clean-t0"],
        flags["clean-duration"],
        flags["fs"],
        conn_out=raw_data_out
    )
    data_generator = mp.Process(target=raw_data_buffer)

    preprocess_buffer = InputDataBuffer(
        flags["batch_size"],
        flags["chanslist"],
        flags["clean-kernel"],
        flags["clean-stride"],
        flags["fs"],
        ppr_file=flags["ppr-file"],
        conn_in=raw_data_in,
        conn_out=preproc_data_out
    )
    preprocessor = mp.Process(target=preprocess_buffer)

    client_buffer = AsyncInferenceClient(
        flags["url"],
        flags["model_name"],
        flags["model_version"],
        conn_in=preproc_data_in,
        conn_out=infer_result_out
    )
    client = mp.Process(target=client_buffer)

    postprocess_buffer = PostProcessBuffer(
        flags["ppr_file"],
        conn_in=infer_result_out
    )
    postprocessor = mp.Process(target=postprocess_buffer)

    processes = [
        data_generator,
        preprocessor,
        client,
        postprocessor
    ]

    # start all of our processes inside try/except
    # block so that we can shut everything down if
    # there are any errors (including keyboard
    # interrupt to end things)
    try:
        for process in processes:
            process.start()
        while True:
            # TODO: add something here that streams data
            # to a bokeh app or something for visualization,
            # or maybe even add a server process that listens
            # for updates to the model
            continue

    except Exception as e:
        for process in processes:
            if process.is_alive():
                process.stop()
                process.join()
        raise e


if __name__ == "__main__":
    parser = build_parser(get_parser())
    flags = parser.parse_args()
    main(flags)
