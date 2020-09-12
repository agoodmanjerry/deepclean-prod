import multiprocessing as mp
import numpy as np
import time
import pickle
from functools import partial
import sys
sys.path.insert(0, "/opt/conda/pkgs/nds2-client-0.16.6-hd02d5f2_0/libexec/nds2-client/modules/nds2-1_6/")

from bokeh.layouts import row, column
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.server.server import Server
from bokeh.palettes import Category10_4 as palette
from bokeh.transforms import cumsum

from inference_sim import build_simulation, StreamingMetric
from deepclean_prod import signal


SAMPLES_IN_LINE = 2000
SUB_SAMPLE = 2
NUM_SAMPLES_EXTEND = 800

streaming_metrics = {
    "throughput": StreamingMetric(0.95),
    "latency": StreamingMetric(0.95)
}

data_streams = {
    "pred": np.array([]),
    "target": np.array([]) 
}


def get_data():
    '''
    get data from simulation pipe and update both
    streaming tensors and latency/throughput estimates
    '''
    global data_streams
    global streaming_metrics
    global sources

    global start_time
    global processes
    if start_time is None:
        start_time = time.time()
        for process in processes:
            process.start()

    # return model prediction, target channel,
    # the timestamp at which processing was finished,
    prediction, target, output_tstamp = out_pipe.get()
    batches = streaming_metrics["latency"].samples_seen + 1
    samples_seen = batches*flags["batch_size"]

    # we'll measure latency from the *last sample* of the *first frame*
    # we can calculate this time as:
    #     start_time
    time_sample_generated = start_time
    #     plus time delta to the start of this batch
    time_sample_generated += flags["clean_stride"]*(samples_seen - 1)
    #     plus the length of time of a single frame
    time_sample_generated += flags["clean_kernel"]
    print(time_sample_generated)

    latency = output_tstamp - time_sample_generated
    latency = int(latency*10**6)
    streaming_metrics["latency"].update(latency)

    throughput = samples_seen / (output_tstamp - start_time)
    streaming_metrics["throughput"].update(throughput)

    # append data to existing data stream
    target = bandpass(target)
    prediction = target - prediction

    data_streams["pred"] = np.concatenate([data_streams["pred"], prediction])
    data_streams["target"] = np.concatenate([data_streams["target"], target])

    # update our throughput/latency plots
    std_latency = max(np.sqrt(streaming_metrics["latency"].var), 1e-9)
    std_throughput = np.sqrt(streaming_metrics["throughput"].var)
    ellipse_x = np.linspace(
        streaming_metrics["latency"].mean - std_latency,
        streaming_metrics["latency"].mean + std_latency,
        20
    )
    ellipse_y = std_latency**2 - (ellipse_x - streaming_metrics["latency"].mean)**2
    ellipse_y = np.clip(ellipse_y, 0, np.inf)
    ellipse_y = np.sqrt(ellipse_y)
    ellipse_y *= std_throughput / std_latency
    upper_ellipse = streaming_metrics["throughput"].mean + ellipse_y
    lower_ellipse = streaming_metrics["throughput"].mean - ellipse_y

    new_data = {
        "circle": {
            metric_name: metric.mean for metric_name, metric in streaming_metrics.items()
        },
        "patches": {
            "latency": np.concatenate([ellipse_x, ellipse_x[::-1]]),
            "throughput": np.concatenate([upper_ellipse, lower_ellipse[::-1]])
        }
    }
    for source_name, data in new_data.items():
        for metric_name, x in data.items():
            values = sources[source_name].data[metric_name]
            values[-1] = x
            sources[source_name].data[metric_name] = values

    new_queue_data = sources["queue"].data.copy()
    new_pie_data = sources["pie"].data.copy()
    for i, buffer in enumerate(buffers[1:]):
        qsize = buffer.pipe_out.qsize()
        new_queue_data["xs"][i].append(batches)
        new_queue_data["ys"][i].append(qsize)
        for axis in ["xs", "ys"]:
            new_queue_data[axis][i] = new_queue_data[axis][i][-200:]

        new_pie_data["value"][i] = int(buffer.latency*10**6)

    total_latency = sum(new_pie_data["value"])
    new_pie_data["angle"] = [v*2*np.pi / total_latency for v in new_pie_data["value"]]

    sources["queue"].data = new_queue_data
    sources["pie"].data = new_pie_data



def update_line():
    '''
    update our line plots of data up to the maximum
    number of allowed points
    '''
    global data_streams
    global sources

    new_data = sources["line"].data.copy()
    for y in ["pred", "target"]:
        count = len(new_data[y])
        if count >= SAMPLES_IN_LINE:
            # if we're already at or over, cut back
            # until we're the max amount under then extend
            num_over = count - SAMPLES_IN_LINE
            num_trim = num_over + NUM_SAMPLES_EXTEND
            new_data[y] = new_data[y][num_trim:]
            count = len(new_data[y])

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
    sources["line"].data = new_data


def close_shop():
    for buffer in buffers:
        buffer.stop()
    for process in processes:
        if process.is_alive():
            process.join()


def application(doc):
    p1 = figure(
        title="Signal Trace",
        plot_height=400,
        plot_width=600,
        tools=""
    )
    p1.line("x", "pred", line_color="red", source=sources["line"], legend_label="Cleaned")
    p1.line("x", "target", line_color="blue", source=sources["line"], legend_label="Raw")

    p2 = figure(
        title="Pipeline Throughput vs. Latency",
        plot_height=400,
        plot_width=400,
        y_axis_label="Throughput (Frames / s)",
        x_axis_label="Latency (us)",
        tools=""
    )
    p2.circle(
        x="latency",
        y="throughput",
        fill_color="color",
        fill_alpha=0.9,
        line_color="color",
        line_alpha=0.95,
        legend_group="label",
        source=sources["circle"]
    )
    p2.patches(
        xs="latency",
        ys="throughput",
        fill_alpha=0.4,
        fill_color="color",
        line_alpha=0.9,
        line_color="color",
        legend_group="label",
        source=sources["patches"]
    )

    p3 = figure(
        title="Latency Breakdown",
        plot_height=400,
        plot_Width=400
        tools="",
        tooltips=[("@label", "@value us")]
    )
    p3.wedge(
        x=0,
        y=1,
        radius=0.4,
        start_angle=cumsum("angle", include_zero=True),
        end_angle=cumsum("angle"),
        line_color="white",
        fill_color="color",
        legend_group="label",
        source=sources["pie"]
    )

    p4 = figure(
        title="Queue Size",
        plot_height=400,
        plot_width=900,
        y_axis_label="Q Size",
        x_axis_label="Step"
    )
    p4.multi_line(xs="xs", ys="ys", line_color="color", legend_group="label", source=sources["queue"])

    layout = row(p1, p2, p3)
    layout = column(layout, p4)
    doc.add_root(layout)
    doc.add_periodic_callback(update_line, 40)
    doc.add_periodic_callback(get_data, 20)


server = Server({'/': application})
server.start()

if __name__ == '__main__':
    from parse_utils import get_client_parser
    parser = get_client_parser()
    flags = vars(parser.parse_args())

    with open(flags["ppr_file"], "rb") as f:
        ppr = pickle.load(f)
        bandpass = partial(
            signal.bandpass,
            fs=flags["fs"],
            fl=ppr["filt_fl"],
            fh=ppr["filt_fh"],
            order=ppr["filt_order"]
        )

    buffers, out_pipe = build_simulation(flags)
    processes = [mp.Process(target=buffer) for buffer in buffers]

    glyph_data = {
        "line": {
            "x": [],
            "pred": [],
            "target": []
        },
        "circle": {
            "latency": [0],
            "throughput": [0],
            "label": ["1"], # TODO: make number of concurrent models
            "color": ["#65a1c2"]
        },
        "patches": {
            "latency": [[]],
            "throughput": [[]],
            "color": ["#65a1c2"],
            "label": ["1"]
        },
        "queue": {
            "xs": [[] for _ in range(3)],
            "ys": [[] for _ in range(3)],
            "color": palette[:3],
            "label": ["preproc", "inference", "postproc"]
        },
        "pie": {
            "angle": [2*np.pi / 3 for _ in range(3)],
            "value": [0 for _ in range(3)],
            "color": palette[:3],
            "label": ["preproc", "inference", "postproc"]
        }
    }
    sources = {
        glyph: ColumnDataSource(data) for glyph, data in glyph_data.items()
    }
    try:
        start_time = None
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

    except Exception as e:
        close_shop()
        raise e
