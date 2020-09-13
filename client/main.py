from collections import defaultdict
from functools import partial
import multiprocessing as mp
import numpy as np
import pickle
import queue
import re
import sys
import time

from bokeh.layouts import row, column
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.server.server import Server
from bokeh.palettes import Category10_4 as palette
from bokeh.transform import cumsum

sys.path.insert(0, "/opt/conda/pkgs/nds2-client-0.16.6-hd02d5f2_0/libexec/nds2-client/modules/nds2-1_6/")
from inference_sim import build_simulation, StreamingMetric
from deepclean_prod import signal


# TODO: include in flags and pass to app
SAMPLES_IN_LINE = 2000
SUB_SAMPLE = 2
NUM_SAMPLES_EXTEND = 800


class VizApp:
    def __init__(self, buffers, q, warm_up_batches=50):
        self.buffers = buffers
        self.q = q

        self.streaming_metrics = defaultdict(StreamingMetric)
        self.data_streams = defaultdict(lambda : np.array([]))
        self.latency_breakdown = defaultdict(lambda : defaultdict(StreamingMetric))

        glyph_data = {
            "line": {
                "x": [],
                "cleaned": [],
                "raw": []
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
            "pie": {
                "angle": [2*np.pi / len(buffers) for _ in range(len(buffers))],
                "value": [0 for _ in range(len(buffers))],
                "color": palette[:len(buffers)],
                "label": [buff.__name__ for buff in buffers]
            }
        }
        self.sources = {
            glyph: ColumnDataSource(data) for glyph, data in glyph_data.items()
        }
        self.warm_up_batches = warm_up_batches
        self.layout = None
        self.start_time = None

    def get_data(self):
        '''
        get data from simulation pipe and update both
        streaming tensors and latency/throughput estimates
        '''
        num_iters = self.warm_up_batches if self.start_time is None else 1
        for _ in range(num_iters):
            prediction, target, output_tstamp, batch_start_time = self.q.get()
        if self.start_time is None:
            self.start_time = batch_start_time

        # append data to existing data stream
        # TODO: make bandpass in init
        target = bandpass(target)
        prediction = target - prediction

        for stream, y in zip(["cleaned", "raw"], [prediction, target]):
            self.data_streams[stream] = np.append(self.data_streams[stream], y)

        if len(self.data_streams["x"]) == 0:
            last_x = self.start_time - 1/flags["fs"]
        else:
            last_x = self.data_streams["x"][-1]
        new_x = last_x + np.arange(1, len(target)+1)/flags["fs"]
        self.data_streams["x"] = np.append(self.data_streams["x"], new_x)

        latency = int((output_tstamp - batch_start_time)*10**3)
        self.streaming_metrics["latency"].update(latency)

        # TODO: don't use flags, pass to __init__
        batches = self.streaming_metrics["latency"].samples_seen
        samples_seen = batches*flags["batch_size"]

        # TODO: is there a better calc for this?
        throughput = samples_seen / (output_tstamp - self.start_time)
        self.streaming_metrics["throughput"].update(throughput)

        # update our per-func latency breakdowns
        for buff in self.buffers:
            for i in range(20):
                try:
                    func, latency = buff.latency_q.get_nowait()
                except queue.Empty:
                    break
                self.latency_breakdown[buff][func].update(latency)

    def update_throughput_latency_plot(self):
        mean_latency = self.streaming_metrics["latency"].mean
        mean_throughput = self.streaming_metrics["throughput"].mean

        std_latency = max(np.sqrt(self.streaming_metrics["latency"].var), 1e-9)
        std_throughput = np.sqrt(self.streaming_metrics["throughput"].var)

        ellipse_x = np.linspace(
            mean_latency-std_latency, mean_latency+std_latency, 20
        )
        ellipse_y = std_latency**2 - (ellipse_x - mean_latency)**2
        ellipse_y = np.clip(ellipse_y, 0, np.inf)
        ellipse_y = np.sqrt(ellipse_y)
        ellipse_y *= std_throughput / std_latency

        upper_ellipse = mean_throughput + ellipse_y
        lower_ellipse = mean_throughput - ellipse_y

        new_data = {
            "circle": {
                "latency": mean_latency,
                "throughput": mean_throughput
            },
            "patches": {
                "latency": np.append(ellipse_x, ellipse_x[::-1]),
                "throughput": np.append(upper_ellipse, lower_ellipse[::-1])
            }
        }
        for source_name, data in new_data.items():
            for metric_name, x in data.items():
                values = self.sources[source_name].data[metric_name]
                values[-1] = x
                self.sources[source_name].data[metric_name] = values


    def update_latency_breakdown_plot(self):
        new_pie_data = self.sources["pie"].data.copy()
        for i, buff in enumerate(self.buffers):
            latencies = self.latency_breakdown[buff]
            latency = 0
            for func_name, l in latencies.items():
                if func_name not in buff._LATENCY_WHITELIST:
                    latency += l.mean
            new_pie_data["value"][i] = int(latency*10**6)

        total_latency = sum(new_pie_data["value"])
        if total_latency == 0:
            return

        new_pie_data["angle"] = [v*2*np.pi / total_latency for v in new_pie_data["value"]]
        self.sources["pie"].data = new_pie_data

    def update_signal_trace_plot(self):
        '''
        update our line plots of data up to the maximum
        number of allowed points
        '''
        new_data = self.sources["line"].data.copy()
        for stream_name, stream in self.data_streams.items():
            plotted_array = new_data[stream_name]
            if len(plotted_array) >= SAMPLES_IN_LINE:
                # if we're already at or over, cut back
                # until we're the max amount under then extend
                num_over = len(plotted_array) - SAMPLES_IN_LINE
                num_trim = num_over + NUM_SAMPLES_EXTEND
                plotted_array = plotted_array[num_trim:]

            # avoid overflowing
            num_extend = min(
                NUM_SAMPLES_EXTEND, SAMPLES_IN_LINE - len(plotted_array)
            )
            plotted_array.extend(stream[:num_extend])
            new_data[stream_name] = plotted_array
            self.data_streams[stream_name] = stream[num_extend:]

        # update data source
        self.sources["line"].data = new_data

    def build_layout(self):
        p1 = figure(
            title="Signal Trace",
            plot_height=400,
            plot_width=600,
            toolbar_location=None
        )

        for y, color in zip(["cleaned", "raw"], palette):
            p1.line(
                "x",
                y,
                line_color=color,
                line_alpha=0.8,
                source=self.sources["line"],
                legend_label=y.title()
            )

        p2 = figure(
            title="Pipeline Throughput vs. Latency",
            plot_height=400,
            plot_width=400,
            y_axis_label="Throughput (Frames / s)",
            x_axis_label="Latency (ms)",
            toolbar_location=None
        )
        p2.circle(
            x="latency",
            y="throughput",
            fill_color="color",
            fill_alpha=0.9,
            line_color="color",
            line_alpha=0.95,
            legend_group="label",
            source=self.sources["circle"]
        )
        p2.patches(
            xs="latency",
            ys="throughput",
            fill_alpha=0.4,
            fill_color="color",
            line_alpha=0.9,
            line_color="color",
            legend_group="label",
            source=self.sources["patches"]
        )

        p3 = figure(
            title="Latency Breakdown",
            plot_height=400,
            plot_width=400,
            tools="hover",
            toolbar_location=None,
            x_range=(-0.5, 1.0),
            tooltips=[("@label", "@value us")]
        )
        p3.axis.visible = False
        p3.axis.axis_label = None
        p3.grid.grid_line_color=None

        p3.wedge(
            x=0,
            y=1,
            radius=0.5,
            start_angle=cumsum("angle", include_zero=True),
            end_angle=cumsum("angle"),
            line_color="white",
            fill_color="color",
            legend_field="label",
            source=self.sources["pie"]
        )

        self.layout = row(p1, p2, p3)

    def __call__(self, doc):
        if self.layout is None:
            self.build_layout()
        doc.add_root(self.layout)

        # update our data frequently
        doc.add_periodic_callback(self.get_data, 20)

        # TODO: should the signal traces be updated
        # at a representative cadence?
        for attr in self.__dir__():
            if re.match("update_.+_plot", attr):
                func = getattr(self, attr)
                doc.add_periodic_callback(func, 100)

    def close_shop(self, processes):
        for buffer in self.buffers:
            buffer.stop()
        for process in processes:
            if process.is_alive():
                process.join()

    def run(self, server, data_generator):
        processes = []
        for buff in [data_generator] + buffers:
            p = mp.Process(target=buff)
            p.start()
            processes.append(p)

        try:
            server.io_loop.add_callback(server.show, "/")
            server.io_loop.start()
        except Exception as e:
            self.close_shop(processes)
            raise e


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
    data_generator = buffers.pop(0)
    application = VizApp(buffers, out_pipe)

    server = Server({'/': application})
    server.start()
    application.run(server, data_generator)

