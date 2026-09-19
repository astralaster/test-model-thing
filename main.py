import argparse
import glob
import itertools
import multiprocessing
import os
import random
import signal
import sys
import time
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util

class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def __call__(self, x: mx.array): return self.embed(x)

class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def __call__(self, x: mx.array): return self.decode(x), mx.sigmoid(self.stop(x))

class Layer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
        self.decay = mx.zeros((dim, ))
        self.states = mx.zeros((dim, ))

        self.decaytrace = mx.zeros((dim, ))
        self.embedtrace = mx.zeros((256, dim))
        
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias = False)
        self.silu = nn.SiLU()

        self.freeze(keys = ["states", "decaytrace", "embedtrace"], recurse = False)

    def __call__(self, enc: mx.array, x: mx.array, dummy: mx.array):
        decay = mx.sigmoid(self.decay)
        state = (decay * self.states) + enc + dummy

        return x + self.silu(self.weights(self.norm(state))), state, decay

class Model(nn.Module):
    def __init__(self, dim: int, layers: int, temp: float, lr: float, accum: int = 1):
        super().__init__()
        self.dim = dim
        self.layercount = layers
        self.temp = temp
        self.accum_k = max(1, int(accum))

        object.__setattr__(self, "accum", None)
        self.accum_n = 0
        self.opt_steps = 0

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

        self.layers = [Layer(dim) for _ in range(layers)]
        self.optimizer = opt.AdamW(learning_rate = lr)
        self.optimizer.init(self.trainable_parameters())

        self.opt_step = mx.compile(self._apply_gradients)

    def _apply_gradients(self, grads, params, state):
        self.optimizer._state = state

        params = self.optimizer.apply_gradients(grads, params)
        return params, self.optimizer._state

    def _accumulate(self, grads, params):
        if self.accum_k == 1:
            self.opt_steps += 1
            return self.opt_step(grads, params, self.optimizer._state)

        scale = 1.0 / self.accum_k

        if self.accum is None: object.__setattr__(self, "accum", util.tree_map(lambda g: g * scale, grads))
        else: object.__setattr__(self, "accum", util.tree_map(lambda a, g: a + g * scale, self.accum, grads))

        mx.eval(self.accum)
        self.accum_n += 1

        if self.accum_n < self.accum_k: return params, self.optimizer._state

        grads = self.accum
        object.__setattr__(self, "accum", None)
        self.accum_n = 0
        self.opt_steps += 1

        return self.opt_step(grads, params, self.optimizer._state)

    def sample(self, output: mx.array):
        probs = mx.softmax(output)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8)) / mx.log(mx.array(256.0))

        temp = mx.maximum(0.1, self.temp * (1.0 - self.temp * entropy)).item()
        return mx.random.categorical(output / temp)

    def evaluate(self): mx.eval(*[layer.states for layer in self.layers])

    def reset(self):
        for layer in self.layers:
            layer.decay = mx.zeros((self.dim, ))
            layer.states = mx.zeros((self.dim, ))

            layer.decaytrace = mx.zeros((self.dim, ))
            layer.embedtrace = mx.zeros((256, self.dim))

        self.evaluate()

    def step(self, c: mx.array, dummies: mx.array | None = None, frozen: bool = False):
        if dummies is None: dummies = [mx.zeros((self.dim, )) for _ in range(self.layercount)]

        enc = self.encoder(c)
        x = enc
            
        states, decays = [], []

        for i, layer in enumerate(self.layers):
            x, state, decay = layer(enc, x, dummies[i])
            if frozen: layer.states = mx.stop_gradient(state)

            states.append(state)
            decays.append(decay)

        return (x, states, decays), self.decoder(x)

    def __call__(self, currb: int, nextb: int | None, end: bool, frozen: bool):
        c = mx.array(currb)

        if frozen:
            _, (output, stop) = self.step(c, frozen = True)

            self.evaluate()
            return self.sample(output).item(), stop.item()

        p = self.trainable_parameters()

        def fwd(params, dummies: list[mx.array]):
            self.update(params)

            (x, states, decays), (output, stop) = self.step(c, dummies)

            loss = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4))
            if nextb is not None:
                n = mx.array(nextb)
                tgt = mx.stop_gradient(self.encoder(n))

                loss = loss + mx.mean(mx.square(x - tgt))
                loss = loss - output[n] + mx.logsumexp(output)

                loss = loss + mx.mean(mx.square(stop - mx.array([1.0 if end else 0.0])))
                
            # loss = variance loss + pred mse loss + crossentropy loss + stop mse loss
            return loss, (states, decays, output, stop)

        (_, (states, decays, output, stop)), (grads, dlds_s) = mx.value_and_grad(
            fwd, argnums = (0, 1)
        )(p, [mx.zeros((self.dim, )) for _ in range(self.layercount)])

        self.update(p)

        for i, layer in enumerate(self.layers):
            dlds = dlds_s[i]

            embedtrace = (layer.embedtrace * decays[i]) + (mx.arange(256) == c)[:, None].astype(mx.float32)
            grads["encoder"]["embed"]["weight"] += dlds * (layer.embedtrace * decays[i])
            
            decaytrace = (decays[i] * layer.decaytrace) + (decays[i] * (1.0 - decays[i]) * layer.states)
            grads["layers"][i]["decay"] = dlds * decaytrace

            layer.states = mx.stop_gradient(states[i])

            layer.decaytrace = mx.stop_gradient(decaytrace)
            layer.embedtrace = mx.stop_gradient(embedtrace)

        params, state = self._accumulate(grads, p)

        self.optimizer._state = state
        self.update(params)
        mx.eval(self.parameters())

        return self.sample(output).item(), stop.item()

    def save(self, path: str):
        data = {}
        for k, v in util.tree_flatten(self.parameters()): data[f"m.{k}"] = v
        for k, v in util.tree_flatten(self.optimizer.state): data[f"o.{k}"] = v

        for i, layer in enumerate(self.layers):
            data[f"state.{i}"] = layer.states
            data[f"decaytrace.{i}"] = layer.decaytrace
            data[f"embedtrace.{i}"] = layer.embedtrace

        if self.accum is not None:
            for k, v in util.tree_flatten(self.accum): data[f"g.{k}"] = v

        data["acc.n"] = mx.array([self.accum_n], dtype = mx.int32)
        data["acc.steps"] = mx.array([self.opt_steps], dtype = mx.int32)

        tmp = os.path.join(os.path.dirname(path), 'temporary-' + os.path.basename(path))
        mx.save_safetensors(tmp, data)
        os.replace(tmp, path)

    def load(self, path: str):
        if not os.path.exists(path): return

        data = mx.load(path)
        model, opts = {}, {}
        accum = {}

        params = set(dict(util.tree_flatten(self.parameters())).keys())
        trainable = set(dict(util.tree_flatten(self.trainable_parameters())).keys())

        for k, v in data.items():
            if k.startswith("m."):
                key = k[2:]
                if key in params: model[key] = v
            elif k.startswith("o."):
                key = k[2:]
                base = key[:-2] if key.endswith((".m", ".v")) else key
                if base in trainable or base in ("step", "learning_rate"): opts[key] = v
            elif k.startswith("state."): self.layers[int(k.split('.')[1])].states = v
            elif k.startswith("decaytrace."): self.layers[int(k.split('.')[1])].decaytrace = v
            elif k.startswith("embedtrace."): self.layers[int(k.split('.')[1])].embedtrace = v
            elif k.startswith("g."): accum[k[2:]] = v
            elif k == "acc.n": self.accum_n = int(v.item())
            elif k == "acc.steps": self.opt_steps = int(v.item())

        if model: self.update(util.tree_unflatten(list(model.items())))

        if accum: object.__setattr__(self, "accum", util.tree_unflatten(list(accum.items())))

        if opts:
            self.optimizer.state = util.tree_unflatten(list(opts.items()))

        self.optimizer.init(self.trainable_parameters())

    def count(self) -> int:
        per_layer = self.dim * self.dim + 3 * self.dim
        return 256 * self.dim + self.layercount * per_layer + 256 * self.dim + 256 + self.dim + 1

class Runtime:
    def __init__(self, path: str, threshold: float, accum: int = 1, sync_every: int = 0, rank: int = 0, workers: int = 1, distributed = None, **kwargs):
        self.model = Model(accum = accum, **kwargs)
        self.path = path
        self.threshold = threshold
        self.sync_every = sync_every
        self.rank = rank
        self.workers = workers
        self.dist = distributed

        self.step = 0
        self.prevtime = None

    def save(self):
        self.step += 1
        if self.step % 500 == 0 and (self.dist is None or self.rank == 0): self.model.save(self.path)

    def call(self, c: int, n: int | None, end: bool, save: bool, frozen: bool):
        before = self.model.opt_steps
        outputs = self.model(c, n, end, frozen)

        if self.dist is not None and self.sync_every and self.model.opt_steps != before and self.model.opt_steps % self.sync_every == 0:
            self.dist.sync(self.model)

        if save: self.save()
        return outputs

    def write(self, b: int):
        if self.rank: return

        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, save: bool, frozen: bool):
        while True:
            text = input(f'\n[{self.now()} | {0 if self.prevtime is None else time.time() - self.prevtime:.4f}s]\nUser >> ')
            self.prevtime = time.time()

            data = (text + '\n').encode('utf-8')
            
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                b, _ = self.call(c, n, i == len(data) - 2, save, frozen)

            print(f'\n[{self.now()}]\nModel >> ', end = '', flush = True)

            b = data[-1]
            while True:
                b, stop = self.call(b, None, False, save, frozen)
                self.write(b)

                if stop > self.threshold:
                    print()
                    break

    def train(self, save: bool, frozen: bool, dataset: str, start: int = 0, end: int | None = None):
        files = sorted(glob.glob(dataset, recursive = True))

        if not files:
            raise FileNotFoundError(
                f'Could not find training files with the following glob: {dataset!r}. Try downloading a dataset first.'
            )

        if self.dist is not None: random.Random(0).shuffle(files)
        else: random.shuffle(files)

        sizes = [os.path.getsize(file) for file in files]

        while True:
            base = 0
            stop = False

            for file, size in zip(files, sizes):
                if stop: break

                with open(file, 'rb') as f:
                    pos = 0

                    for data in f:
                        gstart = base + pos
                        pos += len(data)

                        if end is not None and gstart >= end:
                            stop = True
                            break

                        if gstart + len(data) <= start: continue
                        if len(data) < 2: continue

                        for i, (c, n) in enumerate(itertools.pairwise(data)):
                            b, _ = self.call(c, n, i == len(data) - 2, save, frozen)
                            self.write(b)

                base += size

    def distributed_run(self, dataset: str, save: bool, frozen: bool, start: int, end: int):
        self.model.load(self.path)

        self.dist.setup(self.model)
        self.dist.broadcast(self.model)

        print(f'[worker {self.rank}] parameters: {self.model.count():,}', flush=True)

        try:
            self.train(save, frozen, dataset, start, end)
        finally:
            if save and self.rank == 0: self.model.save(self.path)

    def now(self): return datetime.now().strftime('%d/%m/%Y, %H:%M:%S')

    def __call__(self, mode: str, dataset: str, save: bool, frozen: bool):
        self.model.load(self.path)
        print()

        try:
            match mode:
                case 'train': self.train(save, frozen, dataset)
                case 'chat': self.chat(save, frozen)

        finally:
            if save: self.model.save(self.path)

class Distributed:
    def __init__(self, rank: int, workers: int, buffer, barrier):
        self.rank = rank
        self.workers = workers
        self.buffer = buffer
        self.barrier = barrier

        self.view = memoryview(buffer)
        self.bytes = self.view.cast('B')

        self.metas = None
        self.total = None

    def setup(self, model: Model):
        leaves = util.tree_flatten(model.trainable_parameters())

        self.metas = [(k, v.shape, v.size) for k, v in leaves]
        self.total = sum(size for _, _, size in self.metas)

    def _flat(self, model: Model):
        leaves = util.tree_flatten(model.trainable_parameters())

        flat = mx.concatenate([v.reshape(-1) for _, v in leaves])
        mx.eval(flat)

        return flat

    def _write(self, model: Model):
        flat = self._flat(model)
        offset = self.rank * self.total

        self.bytes[offset * 4:(offset + self.total) * 4] = memoryview(flat).cast('B')

    def _apply(self, model: Model, vec: mx.array):
        params, index = {}, 0

        for k, shape, size in self.metas:
            params[k] = vec[index:index + size].reshape(shape)
            index += size

        model.update(util.tree_unflatten(list(params.items())))
        mx.eval(model.parameters())

    def broadcast(self, model: Model):
        if self.rank == 0: self._write(model)

        self.barrier.wait()
        self._apply(model, mx.array(self.view).reshape(self.workers, self.total)[0])
        self.barrier.wait()

    def sync(self, model: Model):
        self._write(model)

        self.barrier.wait()
        mean = mx.mean(mx.array(self.view).reshape(self.workers, self.total), axis = 0)
        self._apply(model, mean)
        self.barrier.wait()

def worker_main(rank: int, config: dict, buffer, barrier, start: int, end: int):
    distributed = Distributed(rank, config['workers'], buffer, barrier)

    runtime = Runtime(
        path = config['path'], threshold = 0.35, dim = config['dim'], layers = config['layers'],
        temp = config['temp'], lr = config['lr'], accum = config['accum'],
        sync_every = config['sync_every'], rank = rank, workers = config['workers'], distributed = distributed,
    )

    runtime.distributed_run(config['dataset'], config['save'], config['frozen'], start, end)

def run_distributed(config: dict):
    context = multiprocessing.get_context('spawn')
    workers = config['workers']

    template = Model(dim = config['dim'], layers = config['layers'], temp = config['temp'], lr = config['lr'], accum = config['accum'])
    total = sum(v.size for _, v in util.tree_flatten(template.trainable_parameters()))

    print(f'parameters: {template.count():,}')

    files = sorted(glob.glob(config['dataset'], recursive = True))

    if not files:
        raise FileNotFoundError(
            f'Could not find training files with the following glob: {config["dataset"]!r}. Try downloading a dataset first.'
        )

    random.Random(0).shuffle(files)
    total_bytes = sum(os.path.getsize(file) for file in files)
    span = total_bytes // workers

    buffer = context.RawArray('f', total * workers)
    barrier = context.Barrier(workers)

    processes = []

    for rank in range(workers):
        start = rank * span
        end = total_bytes if rank == workers - 1 else (rank + 1) * span

        process = context.Process(target = worker_main, args = (rank, config, buffer, barrier, start, end))
        process.daemon = True
        processes.append(process)

    def terminate(*_): raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    for process in processes: process.start()
    try:
        for process in processes: process.join()
    except KeyboardInterrupt:
        for process in processes: process.terminate()
        for process in processes: process.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'test-model-thing')
    parser.add_argument('path')
    parser.add_argument('mode', choices = ['train', 'chat'])

    parser.add_argument('--frozen', action = 'store_true')
    parser.add_argument('--no-save', action = 'store_false')
    parser.add_argument('--dataset', default = 'wikipedia_clean/**/wiki_*')

    parser.add_argument('--workers', type = int, default = 1)
    parser.add_argument('--accum', type = int, default = 8)
    parser.add_argument('--sync-every', type = int, default = 200)

    args = parser.parse_args()

    if args.mode == 'train' and args.workers > 1:
        config = {
            'path': args.path, 'dataset': args.dataset, 'save': args.no_save, 'frozen': args.frozen,
            'dim': 512, 'layers': 16, 'temp': 0.75, 'lr': 5e-4, 'workers': args.workers,
            'accum': args.accum, 'sync_every': args.sync_every,
        }

        run_distributed(config)
    else:
        runtime = Runtime(
            path = args.path, threshold = 0.35, dim = 512, layers = 16, temp = 0.75, lr = 5e-4,
            accum = 1 if args.mode == 'chat' else args.accum, sync_every = args.sync_every,
        )
        print(f'parameters: {runtime.model.count():,}')

        runtime(args.mode, args.dataset, args.no_save, args.frozen)