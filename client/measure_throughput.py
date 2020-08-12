import os
import multiprocessing as mp
import numpy as np
import time
from abc import abstractmethod
from functools import partial

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
    parser.add_argument(
        "--url",
        help="Server url",
        default="localhost:8001",
        type=str
    )
    parser.add_argument(
        "--model-name",
        help="Name of inference model",
        required=True,
        type=str
    )
    parser.add_argument(
        "--model-version",
        help="Model version to use",
        default=1,
        type=int
    )
    # TODO: can we just get this from the input_shape until we
    # get dynamic batching working?
    parser.add_argument(
        "--batch-size",
        help="Number of windows to infer on at once",
        required=True,
        type=int,
    )

    # viz plotting arguments
    parser.add_argument(
        "--output_dir",
        help="Directory to write output files",
        default="/bokeh",
        type=str
    )
    parser.add_argument(
        "--max_write_arrays",
        help="Maximum number of (output, target) arrays to keep at once",
        default=100,
        type=int
    )
    parser.add_argument(
        "--max_measurements",
        help"Maximum number of latency/timestamp samples to keep at once",
        default=1000,
        type=int
    )
    return parser


class MaxablePipe:
    '''
    Cheap wrapper object to be able to leverage the speed of
    pipes but while keeping loose track of how much data is
    being put into queues. Just in case the data generation
    process gets way ahead of the rest of the pipeline, this
    can stop us from overloading the system memory
    '''
    def __init__(self, maxsize=None):
        self.conn_in, self.conn_out = mp.Pipe(duplex=False)
        self.maxsize = maxsize
        self._size = 0

    @property
    def full(self):
        self._size >= self.maxsize

    def put(self, x, timeout=None):
        start_time = time.time()
        while self.full:
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    raise RuntimeError
        self.conn_out.send(x)
        self._size += 1

    def get(self):
        x = self.conn_in.recv()
        self._size -= 1
        return x

    def close(self):
        self.conn_in.close()
        self.conn_out.close()


class StoppableIteratingBuffer:
    def __init__(self, pipe_in=None, pipe_out=None):
        self.pipe_in = pipe_in
        self.pipe_out = pipe_out
        self._stop_event = mp.Event()

    def put(self, x, timeout=None):
        if self.pipe_out is None:
            raise ValueError("Nowhere to put!")
        self.pipe_out.put(x, timeout)

    def get(self):
        if self.pipe_in is None:
            raise ValueError("Nowhere to get!")
        return self.pipe_in.get()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def stop(self):
        self._stop_event.set()

    def __call__(self):
        self.initialize_loop()
        while not self.stopped:
            self.loop()

        # allow for custom subclass cleanup
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
        self.target_channel = channels[0]
        self.channels = sorted(channels[1:])
        super().__init__(**kwargs)

    def initialize_loop(self):
        self.idx = 0
        self.start_time = time.time()

    @property
    def generator_time(self):
        '''
        associate each sample with a time in real time that it
        "should" have been sampled given the start time,
        sample rate, and current_index
        '''
        return self.start_time + self.idx*self.sleep_time

    def loop(self):
        samples = {}
        for channel in self.channels:
            samples[channel] = self.data[channel][self.idx].value

        # TODO: do thread-based write of target here so
        # that it can be read by plotting process with
        # "true" time lag?
        target = self.data[self.target_channel][self.idx].value

        generation_time = self.generator_time
        self.put((samples, target, generation_time))

        # TODO: do some sort of padding rather than
        # just doing wrapping? Would this even be
        # handled here?
        self.idx += 1
        if self.idx == len(self.data[channel]):
            self.start_time += self.idx*self.sleep_time
            self.idx = 0


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
            start = int(i*kernel_stride*fs)
            stop = int(start + kernel_size*fs)
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
        samples, target, gen_time = self.get()

        # make sure that we don't "peek" ahead at
        # data that isn't supposed to exist yet
        while time.time() < gen_time:
            continue

        samples = [[samples[channel]] for channel in self.channels]
        x = np.array(samples, dtype=np.float32)
        return x, target, gen_time

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
        return np.stack([data[:, slc] for slc in self.slices])

    def loop(self):
        # start by reading the next batch of samples
        # TODO: play with numpy to see what's most efficient
        # concat and reshape? read_sensor()[:, None]?
        data, target = [], []
        for i in range(self.num_samples):
            x, y, gen_time = read_sensor()
            if i == 0:
                batch_gen_time = gen_time
            data.append(x)
            target.append(y)

        data = np.hstack(data)
        target = np.array(target)
    
        data = self.preprocess(data)
        data = self.batch(data)
        self.put((data, target, batch_gen_time))


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
        X, y, batch_gen_time = self.get()
        callback=partial(
            self.process_result, target=y, batch_gen_time=batch_gen_time)
    
        self.client_input.set_data_from_numpy(X)
        self.client.async_infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[self.client_input],
            outputs=[self.client_output]
            callback=callback
        )
    
    def process_result(self, target, batch_gen_time, result, error):
        # TODO: add error checking
        data = result.as_numpy(self.client_output.name())
        self.put((data, target, batch_gen_time))


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
        data, target, batch_gen_time = self.get()
        data = self.std*data + self.mean

        # TODO: add overlap calculation, maybe keep track of streaming
        # output tensor to capture overlap from predictions that
        # weren't in this batch?
        data = bandpass(data, self.filt_fl, self.filt_fh, self.filt_order)
        processed_time = time.time()

        # TODO: May need to account for the finite amount of
        # time this takes if `conn_in.recv` is never waiting
        batch_latency = int((processed_time - batch_gen_time)*10**6)

        # send everything to some writer to pass to e.g. viz app
        self.put((data, target, batch_latency, processed_time))


def main(flags):
    # set up output directory
    if not os.path.exists(flags["output_dir"])
        os.makedirs(flags["output_dir"])
    run_name = flags["model_name"] + "-batch_size_" + str(flags["batch_size"])
    output_dir = os.path.join(flags["output_dir"], run_name)

    # TODO: do we need any others?
    for subdir in ["arrays"]:
        subdir = os.path.join(output_dir, subdir)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        elif len(os.listdir(subdir)) > 0:
            for fname in os.listdir(subdir):
                os.remove(os.path.join(subdir, fname))

    # build all the pipes between processes
    # TODO: how will max pipe size affect latency measurements?
    num_samples_per_batch = (
        (flags["clean_stride"]*(flags["batch_size"]-1) + flags["clean_kernel"])*
        flags["fs"]
    )
    max_num_batches = 1000

    raw_data_pipe = MaxablePipe(maxsize=max_num_batches*num_samples_per_batch)
    preproc_pipe = MaxablePipe(maxsize=max_num_batches)
    infer_pipe = MaxablePipe(maxsize=max_num_batches)
    results_pipe = MaxablePipe(maxsize=max_num_batches)

    # pass pipes to iterating buffers to create separate processes
    # for each step in the pipeline. We'll do viz writing in
    # the main process (here)

    # start with data generation process. This is meant to simulate
    # the functionality of the actual sensors from which data will
    # be read, and so wouldn't be used in a real pipeline
    raw_data_buffer = DataGeneratorBuffer(
        flags["chanslist"],
        flags["clean_t0"],
        flags["clean_duration"],
        flags["fs"],
        pipe_out=raw_data_pipe
    )
    data_generator = mp.Process(target=raw_data_buffer)

    # asynchronously read samples from data generation process,
    # accumulate a batch's worth, apply preprocessing, then
    # window chunks into a batch
    preprocess_buffer = InputDataBuffer(
        flags["batch_size"],
        flags["chanslist"][1:],
        flags["clean_kernel"],
        flags["clean_stride"],
        flags["fs"],
        ppr_file=flags["ppr_file"],
        pipe_in=raw_data_pipe,
        pipe_out=preproc_pipe
    )
    preprocessor = mp.Process(target=preprocess_buffer)

    # asynchronously read preprocessed data and submit to
    # triton for model inference
    client_buffer = AsyncInferenceClient(
        flags["url"],
        flags["model_name"],
        str(flags["model_version"]),
        pipe_in=preproc_pipe,
        pipe_out=infer_pipe
    )
    client = mp.Process(target=client_buffer)

    # asynchronously receive outputs from triton and
    # apply postprocessing (denormalizing, bandpass
    # filtering, accumulating across windows)
    postprocess_buffer = PostProcessBuffer(
        flags["ppr_file"],
        pipe_in=infer_pipe,
        pipe_out=results_pipe
    )
    postprocessor = mp.Process(target=postprocess_buffer)

    processes = [
        data_generator,
        preprocessor,
        client,
        postprocessor
    ]
    pipes = [
        raw_data_pipe,
        preproc_pipe,
        infer_pipe,
        results_pipe
    ]

    # start all of our processes inside try/except
    # block so that we can shut everything down if
    # there are any errors (including keyboard
    # interrupt to end things)
    try:
        for process in processes:
            process.start()

        # TODO: instead of files here, should we use
        # mp.connection.Listener and set up a client on
        # the other end?

        # write latency measurements to csv, name it with
        # start time so that we can calculate throughput
        # measurements on the read side however we like
        start_time = time.time()
        i = 0
        lazy_average_latency = 0
        lazy_average_rmse = 0

        csv_rows = ["latency,timestamp"]
        csv_filename = "measurements-" + str(int(start_time*10**6)) + ".csv"
        print("Running demo...")
        while True:
            data, target, latency, tstamp = results_pipe.get()
            # tstamp = str(int(tstamp*10**6))

            # commenting the file writing out until I figure out viz
            # and I/O issues. Just going to print running throughput for now
            i += 1
            elapsed_time = tstamp - start_time
            lazy_throughput = (i*flags["batch_size"]) / elapsed_time
            lazy_average_latency = (i-1)*lazy_average_latency / i + latency / i

            rmse = ((data - target)**2).mean()**(0.5)
            lazy_average_rmse = (i-1)*lazy_average_rmse / i + rmse / i
    
            msg = "\rThroughput: {} samples/second, Latency: {} us, RMSE: {:0.3f}"
            msg.format(
                int(lazy_throughput),
                int(lazy_average_latency*10**6),
                rmse
            )
            print(msg, end="")

            # TODO: add checks to avoid writing to files opened
            # by other processes, particularly for the measurement
            # csv write

            # save out the prediction and the target to a numpy array
            # Create some slack for a reader process to be behind,
            # but cap at some maximum number of files and name them
            # according to us timestamp to know which ones are old and
            # can be deleted
            # arr = np.stack([data, target])
            # array_fnames = sorted(os.listdir(os.path.join(output_dir, "arrays")))
            # if len(array_fnames) > flags["max_write_arrays"]:
            #     # delete the earliest files
            #     os.remove(os.path.join(output_dir, "arrays", array_fnames[0]))
            # np.save(os.path.join(output_dir, "arrays", tstamp), arr)

            # # TODO: does it make more sense to do an append here
            # # and leave the job of deleting stuff to the reading
            # # process? Seems unsafe if the reading process dies
            # # or something
            # row = f"{latency},{timestamp}"
            # if len(csv_rows) >= flags["max_measurements"]:
            #     del csv_rows[1]
            # csv_rows.append(row)
            # csv_string = "\n".join(csv_rows)
            # with open(os.path.join(output_dir, csv_filename), "w") as f:
            #     f.write(csv_string)

    except Exception as e:
        for process in processes:
            if process.is_alive():
                process.stop()
                process.join()
        for pipe in pipes:
            pipe.close()
        raise e


if __name__ == "__main__":
    parser = build_parser(get_parser())
    flags = parser.parse_args()
    main(flags)
