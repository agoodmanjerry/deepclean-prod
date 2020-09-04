import multiprocessing as mp
import os
import pickle
import time
from abc import abstractmethod
from functools import partial

import numpy as np
from gwpy.timeseries import TimeSeriesDict
from deepclean_prod.signal import bandpass, overlap_add

import tritongrpcclient as triton
from tritongrpcclient import model_config_pb2 as model_config
from tritonclientutils import triton_to_np_dtype


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
    '''
    Parent class for callable Process targets that infinitely
    generate data and so have no internal mechanism which
    will terminate when .join is called. Adds a `stop` method
    which will terminate loop and optionally do cleanup, and
    adds methods for putting data in and pulling it out from
    connective pipes between processes
    '''
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
    def __init__(self, channels, t0, duration, fs, **kwargs):
        self.data = TimeSeriesDict.get(
            channels, t0, t0 + duration, nproc=4, allow_tape=True
        )
        self.data.resample(fs)

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
        num_samples = int((kernel_stride*(batch_size-1) + kernel_size)*fs)

        # initialize arrays up front
        self._data = np.empty((len(channels), num_samples))
        self._batch = np.empty((batch_size, len(channels), int(kernel_size*fs)))
        self._target = np.empty((num_samples,))
    
        self.batch_overlap = int(num_samples - fs*kernel_stride*batch_size)
        self.time_offset = kernel_stride*batch_size

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

    def initialize_loop(self):
        for i in range(self.batch_overlap):
            x, y, gen_time = self.read_sensor()
            if i == 0:
                self.batch_start_time = gen_time
            self._data[:, i] = x
            self._target[i] = y

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

        samples = [samples[channel] for channel in self.channels]
        x = np.array(samples, dtype=np.float32)
        return x, target, gen_time

    def preprocess(self):
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
            return (self._data - self.mean) / self.std
        else:
            # TODO: should this be returning a copy?
            # I think this is safe since I don't really
            # do anything to it between now and when
            # it gets assigned to the batch
            return self._data

    def make_batch(self, data):
        '''
        take windows of data at strided intervals and stack them
        '''
        for i, slc in enumerate(self.slices):
            self._batch[i] = data[:, slc]
        # doing a return here in case we decide
        # we need to do a copy, which I think we do
        return self._batch

    def update(self):
        '''
        shift over all the data elements so that we can populate
        the leftovers with the next batch. Also update the
        batch_start_time by a full batch worth of stride times
        '''
        # TODO: does it make sense to do the copy here, since we'll
        # need to be waiting for the next batch of samples to generate
        # anyway?
        self._data[:, :self.batch_overlap] = self._data[:, -self.batch_overlap:]
        self._target[:self.batch_overlap] = self._target[-self.batch_overlap:]
        self.batch_start_time += self.time_offset

    def loop(self):
        # start by reading the next batch of samples
        # TODO: play with numpy to see what's most efficient
        # concat and reshape? read_sensor()[:, None]?
        for i in range(self.batch_overlap, self._data.shape[1]):
            x, y, gen_time = self.read_sensor()
            self._data[:, i] = x
            self._target[i] = y

        data = self.preprocess()
        batch = self.make_batch(data)
        target = self._target.copy()
        self.put((batch, target, self.batch_start_time))

        self.update()


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

        model_metadata = client.get_model_metadata(model_name)
        # TODO: find better way to check version, or even to
        # load specific version
        # assert model_metadata.versions[0] == model_version

        model_input = model_metadata.inputs[0]
        # data_type_name = model_config.DataType.Name(model_input.datatype)
        data_type = model_input.datatype
        # data_type = data_type_name.split("_", maxsplit=1)[1]
        print(data_type)

        model_output = model_metadata.outputs[0]

        self.client_input = triton.InferInput(
            model_input.name, tuple(model_input.shape), data_type
        )
        self.client_output = triton.InferRequestedOutput(model_output.name)
        self.client = client

        self.model_name = model_name
        self.model_version = str(model_version)
        super().__init__(**kwargs)

    def loop(self):
        X, y, batch_gen_time = self.get()
        callback=partial(
            self.process_result, target=y, batch_gen_time=batch_gen_time)
    
        self.client_input.set_data_from_numpy(X.astype('float32'))
        self.client.async_infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[self.client_input],
            outputs=[self.client_output],
            callback=callback
        )
    
    def process_result(self, target, batch_gen_time, result, error):
        # TODO: add error checking
        prediction = result.as_numpy(self.client_output.name())
        self.put((prediction, target, batch_gen_time))


class PostProcessBuffer(StoppableIteratingBuffer):
    def __init__(self, kernel_size, kernel_stride, fs, ppr_file, **kwargs):
        # TODO: should we do a check on the mean and std like we
        # do during preprocessing?
        with open(ppr_file, "rb") as f:
            ppr = pickle.load(f)
            self.uncenter = lambda x: x*ppr["std"] + ppr["mean"]
            self.bandpass = partial(
                bandpass,
                fs=fs,
                fl=ppr["filt_fl"],
                fh=ppr["filt_fh"],
                order=ppr["filt_order"]
            )
        noverlap = int(fs*kernel_size) - int(fs*kernel_stride)
        self.overlap_add = partial(
            overlap_add, noverlap=noverlap, window="boxcar")

        super().__init__(**kwargs)

    def loop(self):
        prediction, target, batch_gen_time = self.get()

        # TOO: include some sort of streaming calculation
        # to keep track of overlap between batches?
        prediction = self.overlap_add(prediction)
        prediction = self.uncenter(prediction)
        prediction = self.bandpass(prediction)

        # measure latency from time at which first
        # frame ended
        completion_time = time.time()
        batch_latency = int((completion_time - batch_gen_time)*10**6)

        # send everything back to main process for handling
        self.put((prediction, target, batch_latency, completion_time))


def build_simulation(flags):
    num_samples_per_batch = (
        (flags["clean_stride"]*(flags["batch_size"]-1) + flags["clean_kernel"])
        *flags["fs"]
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

    # asynchronously read samples from data generation process,
    # accumulate a batch's worth, apply preprocessing, then
    # window chunks into a batch
    preprocess_buffer = InputDataBuffer(
        flags["batch_size"],
        flags["chanslist"][1:],
        flags["clean_kernel"],
        flags["clean_stride"],
        flags["fs"],
        flags["ppr_file"],
        pipe_in=raw_data_pipe,
        pipe_out=preproc_pipe
    )

    # asynchronously read preprocessed data and submit to
    # triton for model inference
    client_buffer = AsyncInferenceClient(
        flags["url"],
        flags["model_name"],
        flags["model_version"],
        pipe_in=preproc_pipe,
        pipe_out=infer_pipe
    )

    # asynchronously receive outputs from triton and
    # apply postprocessing (denormalizing, bandpass
    # filtering, accumulating across windows)
    postprocess_buffer = PostProcessBuffer(
        flags["clean_kernel"],
        flags["clean_stride"],
        flags["fs"],
        flags["ppr_file"],
        pipe_in=infer_pipe,
        pipe_out=results_pipe
    )

    buffers = [
        raw_data_buffer,
        preprocess_buffer,
        client_buffer,
        postprocess_buffer
    ]
    return buffers, results_pipe

