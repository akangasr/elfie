import numpy as np

from elfi.store import OutputPool


class SerializableOutputPool(OutputPool):

    def to_dict(self):
        stores = dict()
        for k, v in self.output_stores.items():
            d = dict()
            for kk, vv in v.items():
                d[kk] = vv.tolist()  # np.array to list
            stores[k] = d
        seed = int(self.seed)  # np.uint32 to int
        return {
                "output_stores": stores,
                "batch_size": self.batch_size,
                "seed": seed,
                }

    @staticmethod
    def from_dict(d):
        p = SerializableOutputPool()
        p.batch_size = int(d["batch_size"])  # string to int
        p.seed = np.uint32(d["seed"])  # int to np.uint32
        stores = dict()
        for k, v in d["output_stores"].items():
            d = dict()
            for kk, vv in v.items():
                d[int(kk)] = np.array(vv)  # string to int, list to np.array
            stores[k] = d
        p.output_stores = stores
        return p

