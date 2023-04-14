__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-26"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import makedirs, getcwd, listdir, remove, system, path
from xml.etree import ElementTree as ET
import pickle
import time
import glob
import dolfin as df
import sys
import json
from oasisx.logging import info_red
import meshio


def mesh_from_file(pth):
    # pth = "/".join(origin.split("/")[:-1])
    mesh = df.Mesh()
    print(path.abspath(pth))
    with df.XDMFFile(pth + "/mesh.xdmf") as infile:
        infile.read(mesh)
    dim = mesh.topology().dim()
    mvc = df.MeshValueCollection("size_t", mesh, dim - 1)
    with df.XDMFFile(pth + "/mf.xdmf") as infile:
        infile.read(mvc, "name_to_read")
    mf = df.cpp.mesh.MeshFunctionSizet(mesh, mvc)
    ds_ = df.ds(subdomain_data=mf, domain=mesh)
    return mesh, mf, ds_


# def mesh_to_file(msh):
#     for cell in msh.cells:
#         if cell.type == "triangle":
#             triangle_cells = cell.data
#         elif  cell.type == "tetra":
#             tetra_cells = cell.data

#     for key in msh.cell_data_dict["gmsh:physical"].keys():
#         if key == "triangle":
#             triangle_data = msh.cell_data_dict["gmsh:physical"][key]
#         elif key == "tetra":
#             tetra_data = msh.cell_data_dict["gmsh:physical"][key]
#     tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})
#     triangle_mesh =meshio.Mesh(points=msh.points,
#                                cells=[("triangle", triangle_cells)],
#                                cell_data={"name_to_read":[triangle_data]})
#     meshio.write("mesh.xdmf", tetra_mesh)

#     meshio.write("mf.xdmf", triangle_mesh)
#     return


def load_json(pth):
    with open(file=pth, encoding="utf-8") as infile:
        some_dict = json.load(infile)
    some_dict["origin"] = pth
    return some_dict


def convert(input):
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iter()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, str):
        return input.encode("utf-8")
    else:
        return input


def parse_command_line():
    commandline_kwargs = {}
    for s in sys.argv[1:]:
        if s.count("=") == 1:
            key, value = s.split("=", 1)
        else:
            raise TypeError(
                (
                    s
                    + " Only kwargs separated with '=' sign "
                    + "allowed. See NSdefault_hooks for a range of "
                    + "parameters. Your problem file should contain "
                    + "problem specific parameters."
                )
            )
        try:
            value = json.loads(value)

        except ValueError:
            if value in (
                "True",
                "False",
            ):  # json understands true/false, but not True/False
                value = eval(value)
            elif "True" in value or "False" in value:
                value = eval(value)
        if isinstance(value, dict):
            value = convert(value)

        commandline_kwargs[key] = value
    return commandline_kwargs


def create_initial_folders(
    folder,
    restart_folder,
    sys_comp,
    tstep,
    # info_red, # TODO: why is this an argument? its imported on top
    scalar_components,
    output_timeseries_as_vector,
    **NS_namespace
):
    """Create necessary folders."""
    info_red("Creating initial folders")
    # To avoid writing over old data create a new folder for each run
    if df.MPI.rank(df.MPI.comm_world) == 0:
        try:
            makedirs(folder)
        except OSError:
            pass

    df.MPI.barrier(df.MPI.comm_world)
    newfolder = path.join(folder, "data")
    if restart_folder:
        newfolder = path.join(newfolder, restart_folder.split("/")[-2])
    else:
        if not path.exists(newfolder):
            newfolder = path.join(newfolder, "1")
        else:
            # previous = listdir(newfolder)
            previous = [f for f in listdir(newfolder) if not f.startswith(".")]
            previous = max(map(eval, previous)) if previous else 0
            newfolder = path.join(newfolder, str(previous + 1))

    df.MPI.barrier(df.MPI.comm_world)
    if df.MPI.rank(df.MPI.comm_world) == 0:
        if not restart_folder:
            # makedirs(path.join(newfolder, "Voluviz"))
            # makedirs(path.join(newfolder, "Stats"))
            # makedirs(path.join(newfolder, "VTK"))
            makedirs(path.join(newfolder, "Timeseries"))
            makedirs(path.join(newfolder, "Checkpoint"))

    tstepfolder = path.join(newfolder, "Timeseries")
    tstepfiles = {}
    comps = sys_comp
    if output_timeseries_as_vector:
        comps = ["p", "u"] + scalar_components

    for ui in comps:
        tstepfiles[ui] = df.XDMFFile(
            df.MPI.comm_world,
            path.join(tstepfolder, ui + "_from_tstep_{}.xdmf".format(tstep)),
        )
        tstepfiles[ui].parameters["rewrite_function_mesh"] = False
        tstepfiles[ui].parameters["flush_output"] = True

    return newfolder, tstepfiles


# FIXME
def save_solution(
    tstep,  # is in NS_parameters
    t,  # is in NS_parameters
    q_,
    q_1,
    folder,  # is in NS_parameters
    newfolder,
    save_step,  # is in NS_parameters
    checkpoint,
    NS_parameters,
    tstepfiles,
    u_,
    u_components,
    scalar_components,  # is in NS_parameters
    output_timeseries_as_vector,  # is in NS_parameters
    constrained_domain,
    AssignedVectorFunction,  # it can be imported from utilities
    killtime,  # is in NS_parameters
    total_timer,
    **NS_namespace
):
    """Called at end of timestep. Check for kill and save solution if required."""
    NS_parameters.update(t=t, tstep=tstep)
    if tstep % save_step == 0:
        save_tstep_solution_h5(
            tstep,  # is in NS_parameters
            q_,
            u_,
            newfolder,
            tstepfiles,
            constrained_domain,
            output_timeseries_as_vector,  # is in NS_parameters
            u_components,
            AssignedVectorFunction,  # it can be imported from utilities
            scalar_components,  # is in NS_parameters
            NS_parameters,
        )

    pauseoasis = check_if_pause(folder)
    while pauseoasis:
        time.sleep(5)
        pauseoasis = check_if_pause(folder)

    killoasis = check_if_kill(folder, killtime, total_timer)
    if tstep % checkpoint == 0 or killoasis:
        save_checkpoint_solution_h5(
            tstep, q_, q_1, newfolder, u_components, NS_parameters
        )

    return killoasis


# FIXME
def save_tstep_solution_h5(
    tstep,  # is in NS_parameters
    q_,
    u_,
    newfolder,
    tstepfiles,
    constrained_domain,
    output_timeseries_as_vector,  # is in NS_parameters
    u_components,
    AssignedVectorFunction,  # it can be imported from utilities
    scalar_components,  # is in NS_parameters
    NS_parameters,
):
    """Store solution on current timestep to XDMF file."""
    timefolder = path.join(newfolder, "Timeseries")
    if output_timeseries_as_vector:
        # project or store velocity to vector function space
        for comp, tstepfile in tstepfiles.items():
            if comp == "u":
                # Create vector function and assigners
                uv = AssignedVectorFunction(u_)

                # Assign solution to vector
                uv()

                # Store solution vector
                tstepfile.write(uv, float(tstep))

            elif comp in q_:
                tstepfile.write(q_[comp], float(tstep))

            else:
                tstepfile.write(tstepfile.function, float(tstep))

    else:
        for comp, tstepfile in tstepfiles.items():
            tstepfile << (q_[comp], float(tstep))

    if df.MPI.rank(df.MPI.comm_world) == 0:
        if not path.exists(path.join(timefolder, "params.dat")):
            f = open(path.join(timefolder, "params.dat"), "wb")
            pickle.dump(NS_parameters, f)


def save_checkpoint_solution_h5(
    tstep, q_, q_1, newfolder, u_components, NS_parameters
):
    """Overwrite solution in Checkpoint folder.

    For safety reasons, in case the solver is interrupted, take backup of
    solution first.

    Must be restarted using the same mesh-partitioning. This will be fixed
    soon. (MM)

    """
    checkpointfolder = path.join(newfolder, "Checkpoint")
    NS_parameters["num_processes"] = df.MPI.size(df.MPI.comm_world)
    if df.MPI.rank(df.MPI.comm_world) == 0:
        if path.exists(path.join(checkpointfolder, "params.dat")):
            system(
                "cp {0} {1}".format(
                    path.join(checkpointfolder, "params.dat"),
                    path.join(checkpointfolder, "params_old.dat"),
                )
            )
        f = open(path.join(checkpointfolder, "params.dat"), "wb")
        pickle.dump(NS_parameters, f)

    df.MPI.barrier(df.MPI.comm_world)
    for ui in q_:
        h5file = path.join(checkpointfolder, ui + ".h5")
        oldfile = path.join(checkpointfolder, ui + "_old.h5")
        # For safety reasons...
        if path.exists(h5file):
            if df.MPI.rank(df.MPI.comm_world) == 0:
                system("cp {0} {1}".format(h5file, oldfile))
        df.MPI.barrier(df.MPI.comm_world)
        ###
        newfile = df.HDF5File(df.MPI.comm_world, h5file, "w")
        newfile.flush()
        newfile.write(q_[ui].vector(), "/current")
        if ui in u_components:
            newfile.write(q_1[ui].vector(), "/previous")
        if path.exists(oldfile):
            if df.MPI.rank(df.MPI.comm_world) == 0:
                system("rm {0}".format(oldfile))
        df.MPI.barrier(df.MPI.comm_world)
    if df.MPI.rank(df.MPI.comm_world) == 0 and path.exists(
        path.join(checkpointfolder, "params_old.dat")
    ):
        system("rm {0}".format(path.join(checkpointfolder, "params_old.dat")))


def check_if_kill(folder, killtime, total_timer):
    """Check if user has put a file named killoasis in folder or if given killtime has been reached."""
    found = 0
    if "killoasis" in listdir(folder):
        found = 1
    collective = df.MPI.sum(df.MPI.comm_world, found)
    if collective > 0:
        if df.MPI.rank(df.MPI.comm_world) == 0:
            remove(path.join(folder, "killoasis"))
            info_red("killoasis Found! Stopping simulations cleanly...")
        return True
    else:
        elapsed_time = float(total_timer.elapsed()[0])
        if killtime is not None and killtime <= elapsed_time:
            if df.MPI.rank(df.MPI.comm_world) == 0:
                info_red(
                    "Given killtime reached! Stopping simulations cleanly..."
                )
            return True
        else:
            return False


def check_if_pause(folder):
    """Check if user has put a file named pauseoasis in folder."""
    found = 0
    if "pauseoasis" in listdir(folder):
        found = 1
    collective = df.MPI.sum(df.MPI.comm_world, found)
    if collective > 0:
        if df.MPI.rank(df.MPI.comm_world) == 0:
            info_red(
                "pauseoasis Found! Simulations paused. Remove "
                + path.join(folder, "pauseoasis")
                + " to resume simulations..."
            )
        return True
    else:
        return False


def check_if_reset_statistics(folder):
    """Check if user has put a file named resetoasis in folder."""
    found = 0
    if "resetoasis" in listdir(folder):
        found = 1
    collective = df.MPI.sum(df.MPI.comm_world, found)
    if collective > 0:
        if df.MPI.rank(df.MPI.comm_world) == 0:
            remove(path.join(folder, "resetoasis"))
            info_red("resetoasis Found!")
        return True
    else:
        return False


def init_from_restart(
    restart_folder,
    sys_comp,
    uc_comp,
    u_components,
    q_,
    q_1,
    q_2,
    tstep,
    **NS_namespace
):
    """Initialize solution from checkpoint files"""
    if restart_folder:
        if df.MPI.rank(df.MPI.comm_world) == 0:
            info_red(
                "Restarting from checkpoint at time step {}".format(tstep)
            )

        for ui in sys_comp:
            filename = path.join(restart_folder, ui + ".h5")
            hdf5_file = df.HDF5File(df.MPI.comm_world, filename, "r")
            hdf5_file.read(q_[ui].vector(), "/current", False)
            q_[ui].vector().apply("insert")
            # Check for the solution at a previous timestep as well
            if ui in uc_comp:
                q_1[ui].vector().zero()
                q_1[ui].vector().axpy(1.0, q_[ui].vector())
                q_1[ui].vector().apply("insert")
                if ui in u_components:
                    hdf5_file.read(q_2[ui].vector(), "/previous", False)
                    q_2[ui].vector().apply("insert")


def merge_visualization_files(newfolder, **namesapce):
    timefolder = path.join(newfolder, "Timeseries")
    # Gather files
    xdmf_files = list(glob.glob(path.join(timefolder, "*.xdmf")))
    xdmf_velocity = [f for f in xdmf_files if "u_from_tstep" in f.__str__()]
    xdmf_pressure = [f for f in xdmf_files if "p_from_tstep" in f.__str__()]

    # Merge files
    for files in [xdmf_velocity, xdmf_pressure]:
        if len(files) > 1:
            merge_xml_files(files)


def merge_xml_files(files):
    # Get first timestep and trees
    first_timesteps = []
    trees = []
    for f in files:
        trees.append(ET.parse(f))
        root = trees[-1].getroot()
        first_timesteps.append(float(root[0][0][0][2].attrib["Value"]))

    # Index valued sort (bypass numpy dependency)
    first_timestep_sorted = sorted(first_timesteps)
    indexes = [first_timesteps.index(i) for i in first_timestep_sorted]

    # Get last timestep of first tree
    base_tree = trees[indexes[0]]
    last_node = base_tree.getroot()[0][0][-1]
    ind = 1 if len(last_node.getchildren()) == 3 else 2
    last_timestep = float(last_node[ind].attrib["Value"])

    # Append
    for index in indexes[1:]:
        tree = trees[index]
        for node in tree.getroot()[0][0].getchildren():
            ind = 1 if len(node.getchildren()) == 3 else 2
            if last_timestep < float(node[ind].attrib["Value"]):
                base_tree.getroot()[0][0].append(node)
                last_timestep = float(node[ind].attrib["Value"])

    # Seperate xdmf files
    new_file = [f for f in files if "_0" in f]
    old_files = [f for f in files if "_" in f and f not in new_file]

    # Write new xdmf file
    base_tree.write(new_file[0], xml_declaration=True)

    # Delete xdmf file
    [remove(f) for f in old_files]
