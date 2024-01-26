import os
from pyTNG import data_interface as _data_interface


def get_sim():
    basepath = "/virgotng/universe/IllustrisTNG/"
    sim_name = "L35n2160TNG"
    sim = _data_interface.TNG50Simulation(os.path.join(basepath, sim_name))
    sim_path = os.path.join(basepath, sim_name, "output")
    return sim, sim_path


def get_redshift(snap_num):
    sim, _ = get_sim()
    z = sim.snap_cat[snap_num].header["Redshift"]
    return z


def scale_factor(z):
    return 1 / (z + 1)
