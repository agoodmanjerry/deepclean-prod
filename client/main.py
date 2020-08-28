import multiprocessing as mp
import numpy as np
import time
import pickle
from functools import partial

from bokeh.layouts import row
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.server.server import Server

from inference_sim import build_simulation
from deepclean_prod import signal


SAMPLES_IN_LINE = 10000
NUM_SAMPLES_EXTEND = 800

streaming_metrics = {
    "batches": 0,
    "mean_latency": 0,
    "mean_throughput": 0,
    "std_latency": 0,
    "std_throughput": 0,
}

data_streams = {
    "pred": np.array([]),
    "target": np.array([]) 
}

line_ds = ColumnDataSource({
    "x": [],
    "pred": [],
    "target": []
})

circ_ds = ColumnDataSource({
    "x": [],
    "y": []
})
patch_ds = ColumnDataSource({
    "x": [],
    "y": []
})


# TODO: add some decay to downweight transient behavior?
def update_running_mean(value, update, N):
    return value + (update-value)/N


def update_running_std(value, update, N, new_mean, old_mean):
    variance = (value + (update-old_mean)*(update-new_mean))/(N-1)
    return np.sqrt(variance)


def get_data():
    '''
    get data from simulation pipe and update both
    streaming tensors and latency/throughput estimates
    '''
    global data_streams
    global streaming_metrics

    # return model prediction, target channel,
    # the timestamp at which processing was finished,
    # and the latency as measured from the *last sample*
    # of the *first batch element*
    prediction, target, tstamp, latency = out_pipe.get()
    streaming_metrics["batches"] += 1

    # append data to existing data stream
    target = bandpass(target)
    prediction = target - prediction

    data_streams["pred"] = np.concatenate([data_streams["pred"], prediction])
    data_streams["target"] = np.concatenate([data_streams["target"], target])

    # update streaming metric estimates
    throughput = (
        streaming_metrics["batches"]*flags["samples_seen"] / 
        (tstamp - start_time)
    )
    updates = {"latency": latency, "throughput": throughput}
    for metric in ["latency", "throughput"]:
        old_mean = streaming_metrics["mean_" + metric]
        update = updates[metric]
        new_mean = update_running_mean(
            old_mean, update, streaming_metrics["batches"])

        old_std = streaming_metrics["std_" + metric]
        new_std = update_running_std(
            old_std,
            update,
            streaming_metrics["batches"],
            new_mean,
            old_mean
        )
        streaming_metrics["mean_"+metric] = new_mean
        streaming_metrics["std_"+metric] = new_std

    new_circle_data = {
        "x": [streaming_metrics["mean_latency"]],
        "y": [streaming_metrics["mean_throughput"]]
    }
    circ_ds.data = new_circle_data

    # plot standard deviations around mean estimates as
    # an ellipse
    ellipse_x = np.linspace(
        streaming_metrics["mean_latency"] - streaming_metrics["std_latency"],
        streaming_metrics["mean_latency"] + streaming_metrics["std_latency"],
        100
    )
    ellipse_eq = 1 - (ellipse_x**2) / (streaming_metrics["std_latency"]**2)
    ellipse_eq *= streaming_metrics["std_throughput"]**2

    upper_ellipse = np.sqrt(ellipse_eq)
    lower_ellipse = -np.sqrt(ellipse_eq)
    new_patch_data = {
        "x": np.concatenate([ellipse_x, ellipse_x[::-1]]),
        "y": np.concatenate([upper_ellipse, lower_ellipse[::-1]])
    }
    patch_ds.data = new_patch_data


def update_line():
    '''
    update our line plots of data up to the maximum
    number of allowed points
    '''
    global data_streams

    new_data = line_ds.data.copy()
    for y in ["pred", "target"]:
        count = len(new_data[y])
        if count >= SAMPLES_IN_LINE:
            # if we're already at or over, cut back
            # until we're the max amount under then extend
            num_over = count - SAMPLES_IN_LINE
            num_trim = num_over + NUM_SAMPLES_EXTEND
            new_data[y] = new_data[y][amount_to_trim:]

        # avoid overflowing
        num_extend = min(NUM_SAMPLES_EXTEND, SAMPLES_IN_LINE-count)
        new_data[y].extend(data_streams[y][:num_extend])
        data_streams[y] = data_streams[y][num_extend:]

    # treat x separately since once we have the number
    # of samples that we want we can stop updating
    # TODO: incorporate accurate time on x axis
    count_x = len(new_data["x"])
    if count_x <= SAMPLES_IN_LINE:
        num_extend = min(NUM_SAMPLES_EXTEND, SAMPLES_IN_LINE - count_x)
        new_data["x"].extend(range(count_x, count_x+num_extend))

    # update data source
    line_ds.data = new_data


def close_shop():
    for buffer in buffers:
        buffer.stop()
    for process in processes:
        if process.is_alive():
            process.join()


def application(doc):
    p1 = figure(plot_height=400, plot_width=600)
    p1.line("x", "pred", line_color="red", source=line_ds)
    p1.line("x", "target", line_color="blue", source=line_ds)

    p2 = figure(plot_height=400, plot_width=400)
    p2.circle("x", "y", source=circ_ds)
    p2.patch("x", "y", source=patch_ds)

    doc.add_root(row(p1, p2))
    doc.add_periodic_callback(update_line, _EXTEND/4)
    doc.add_periodic_callback(get_data, 100)


server = Server({'/': application})
server.start()

if __name__ == '__main__':
    from .parse_utils import get_client_parser
    parser = get_client_parser()
    flags = vars(parser.parse_arguments())

    with open(flags["ppr_file"], "r") as f:
        ppr = pickle.load(f)
        bandpass = partial(
            signal.bandpass,
            fs=flags["fs"],
            fl=ppr_file["filt_fl"],
            fh=ppr_file["filt_fh"],
            order=ppr_file["filt_order"]
        )

    buffers, out_pipe = build_simulation(flags)
    processes = [mp.Process(target=buffer) for buffer in buffers]
    try:
        for process in processes:
            process.start()
        start_time = time.time()

        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

    except Exception as e:
        close_shop()
        raise e
