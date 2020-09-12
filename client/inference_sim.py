import multiprocessing as mp
import os
import pickle
import time
from abc import abstractmethod
from functools import partial
from itertools import cycle

import numpy as np
from gwpy.timeseries import TimeSeriesDict
from deepclean_prod.signal import bandpass, overlap_add

import tritongrpcclient as triton
from tritongrpcclient import model_config_pb2 as model_config
from tritonclientutils import triton_to_np_dtype



class StreamingMetric:
    def __init__(self, decay=None):
        if decay is not None:
            assert 0 < decay and decay <= 1
        self.decay = decay
        self.samples_seen = 0
        self.mean = 0
        self.var = 0

    def update(self, measurement):
        if self.samples_seen == 0:
            self.mean = measurement
        else:
            decay = self.decay or 1./(samples_seen + 1)
            delta = measurement - self.mean
            self.mean += decay*delta
            self.var = (1-decay)*(self.var + decay*delta**2)
        self.samples_seen += 1

        
class MaxablePipe:
    '''
    Cheap wrapper object to be able to leverage the speed of
    pipes but while keeping loose track of how much data is
    being put into queues. Just in case the data generation
    process gets way ahead of the rest of the pipeline, this
    can stop us from overloading the system memory
    '''
    def __init__(self, maxsize=None):
        # TODO: use queues for asynchronous sending and receiving
        # self.conn_in, self.conn_out = mp.Pipe(duplex=False)
        maxsize = int(maxsize) if maxsize is not None else None
        q = mp.Queue(maxsize)
        self.conn_in = q
        self.conn_out = q
        self.maxsize = maxsize
        self._size = 0

    @property
    def full(self):
        return self._size >= self.maxsize

    def get(self):
        return self.conn_in.get()

    def put(self, x, timeout=None):
        return self.conn_out.put(x, timeout=timeout)

#     def put(self, x, timeout=None):
#         start_time = time.time()
#         while self.full:
#             if timeout is not None:
#                 if time.time() - start_time >= timeout:
#                     raise RuntimeError
#         self.conn_out.send(x)
#         self._size += 1
# 
#     def get(self):
#         x = self.conn_in.recv()
#         self._size -= 1
#         return x
# 
#     def close(self):
#         self.conn_in.close()
#         self.conn_out.close()
# 

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


def gwpy_data_generator(data, target_channel, duration, fs):
    target = data.pop(target_channel).value
    channels = list(data.keys())
    data = np.stack([data[channel].value for channel in channels])

    for idx in cycle(range(int(fs*duration))):
        samples = {channel: x for channel, x in zip(channels, data[:, idx])}
        yield samples, target[idx], idx


class DataGeneratorBuffer(StoppableIteratingBuffer):
    def __init__(self, data_generator, **kwargs):
        self.data_generator = iter(data_generator)
        super().__init__(**kwargs)

    def loop(self):
        start_time = time.time()
        samples, target, idx = next(self.data_generator)
        self.put((samples, target))
        end_time = time.time()
        if idx % 2000 == 0:
            print(end_time - start_time)


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
        self.secs_per_sample = 1. / fs
        self.time_since_last = None

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
        self.time_since_last = time.time()
        for i in range(self.batch_overlap):
            x, y = self.read_sensor()
            self._data[:, i] = x
            self._target[i] = y

    def read_sensor(self):
        '''
        read individual samples and return an array of size
        `(len(self.channels), 1)` for hstacking
        '''
        samples, target = self.get()

        # make sure that we don't "peek" ahead at
        # data that isn't supposed to exist yet
        while (time.time() - self.time_since_last) < self.secs_per_sample:
            continue
        self.time_since_last = time.time()

        samples = [samples[channel] for channel in self.channels]
        x = np.array(samples, dtype=np.float32)
        return x, target

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

    def loop(self):
        # start by reading the next batch of samples
        # TODO: play with numpy to see what's most efficient
        # concat and reshape? read_sensor()[:, None]?
        for i in range(self.batch_overlap, self._data.shape[1]):
            x, y = self.read_sensor()
            self._data[:, i] = x
            self._target[i] = y

        data = self.preprocess()
        batch = self.make_batch(data)
        target = self._target.copy()
        self.put((batch, target))

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
        data_type = model_input.datatype
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
        X, y = self.get()
        callback=partial(self.process_result, target=y)
 
        self.client_input.set_data_from_numpy(X.astype('float32'))
        self.client.async_infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[self.client_input],
            outputs=[self.client_output],
            callback=callback
        )
    
    def process_result(self, target, result, error):
        # TODO: add error checking
        prediction = result.as_numpy(self.client_output.name())
        self.put((prediction, target))


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
        prediction, target = self.get()

        # TOO: include some sort of streaming calculation
        # to keep track of overlap between batches?
        prediction = self.overlap_add(prediction)
        prediction = self.uncenter(prediction)
        prediction = self.bandpass(prediction)

        # measure latency from time at which first
        # frame ended
        completion_time = time.time()

        # send everything back to main process for handling
        self.put((prediction, target, completion_time))


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
    data = TimeSeriesDict.get(
        flags["chanslist"],
        flags["clean_t0"],
        flags["clean_t0"]+flags["clean_duration"],
        nproc=4,
        allow_tape=True
    )
    data.resample(flags["fs"])
    data_generator = gwpy_data_generator(
        data, flags["chanslist"][0], flags["clean_duration"], flags["fs"]
    )

    raw_data_buffer = DataGeneratorBuffer(data_generator, pipe_out=raw_data_pipe)

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

