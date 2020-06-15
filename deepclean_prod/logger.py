
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch

# default plotting style
plt.style.use('seaborn-colorblind')
mpl.rc('font', size=15)
mpl.rc('figure', figsize=(8, 5))


class Logger:

    def __init__(self, outdir, metrics):
        self.data_subdir = outdir
        self.metrics = dict([(m, {'steps': [], 'epochs': [], 
                                  'train': [], 'test': []}) for m in metrics])
                
    def update_metric(self, train_metric, test_metric, name, epoch, n_batch, num_batches):
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.data.cpu().numpy()
        if isinstance(test_metric, torch.Tensor):
            test_metric = test_metric.data.cpu().numpy()
            
        step = Logger._step(epoch, n_batch, num_batches)
        self.metrics[name]['train'].append(train_metric)
        self.metrics[name]['test'].append(test_metric)            
        self.metrics[name]['steps'].append(step)
        self.metrics[name]['epochs'].append(step/num_batches)
        
    def log_metric(self, name=None, max_epochs=None):
        out_dir = './{}/metrics'.format(self.data_subdir)
        os.makedirs(out_dir, exist_ok=True)
        
        # If name is not given, log all metrics
        if name is not None:
            train = self.metrics[name]['train']
            test = self.metrics[name]['test']
            steps = self.metrics[name]['steps']
            epochs = self.metrics[name]['epochs']
            
            array = np.vstack((steps, epochs, train, test)).T
            header = 'Step     Epochs    Train     Test'
            np.savetxt('{}/{}.dat'.format(out_dir, name), 
                       array, fmt=('%d, %.2f, %.5f, %.5f'), header=header)
            self._plot_metric(name)
        else:
            for name in self.metrics.keys():
                train = self.metrics[name]['train']
                test = self.metrics[name]['test']
                steps = self.metrics[name]['steps']
                epochs = self.metrics[name]['epochs']
            
                array = np.vstack((steps, epochs, train, test)).T
                header = 'Step     Epochs    Train     Test'
                np.savetxt('{}/{}.dat'.format(out_dir, name), 
                           array, fmt=('%d, %.2f, %.5f, %.5f'), header=header)
                self._plot_metric(name)
            
    def _plot_metric(self, name):
            train = self.metrics[name]['train']
            test = self.metrics[name]['test']
            steps = self.metrics[name]['steps']
            epochs = self.metrics[name]['epochs']
            
            # plot
            fig, ax = plt.subplots()
            ax.plot(steps, train, label='Train')
            ax.plot(steps, test, label='Test')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(name)
            ax.legend()
            
            axtwin = ax.twiny()
            axtwin.plot(epochs, train, alpha=0.)
            axtwin.grid(False)
            axtwin.set_xlabel('Epoch')
            
            # save plot
            out_dir = './{}/metrics'.format(self.data_subdir)
            fig.savefig(f'{out_dir}/{name}.png', dpi=300)
            plt.close()

                        
    def display_status(self, epoch, num_epochs, n_batch, num_batches, 
                       train_metric, test_metric, name, show_epoch=True):
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.data.cpu().numpy()
        if isinstance(test_metric, torch.Tensor):
            test_metric = test_metric.data.cpu().numpy()
        
        if show_epoch:
            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
                epoch,num_epochs, n_batch, num_batches)
                 )
        print('Train {0:}: {1:.4e}, Test {0:}: {2:.4e}'.format(
            name, train_metric, test_metric))

    def save_model(self, model, epoch, n_batch=None):
        out_dir = './{}/models'.format(self.data_subdir)
        os.makedirs(out_dir, exist_ok=True)
        if n_batch is not None:
            torch.save(model.state_dict(), '{}/epoch_{}_batch_{}'.format(out_dir, epoch, n_batch))
        else:
            torch.save(model.state_dict(), '{}/epoch_{}'.format(out_dir, epoch))

    # Private Functionality
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch
    