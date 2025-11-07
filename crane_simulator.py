#!/usr/bin/env python3

from crane_controller import Application, Position, Size, CranePath

from os import isatty
from sys import stdin, stderr
from copy import deepcopy
import argparse
from typing import Generator, Callable
from threading import ThreadError
from math import sqrt
import traceback

import csv

CONTAINER_LENGTH = 6.058
CONTAINER_WIDTH = 2.438
CONTAINER_HEIGHT = 2.591
MARGIN = 0.250

ACCELERATION = [0.2, 0.35, 0.2]
MAX_VELOCITY = [40 / 60, 8 / 60, 32 / 60]
ACCELERATION_RESISTANCE = [1, 9.81, 1]
SIMULATION_SPEED = 10
MASS = 32_000
ATTACH_TIME = 10
DETACH_TIME = 10

DEBUG = False

def translate_distances(pos1: Position, pos2: Position):
    """
    multiply each direction of a distance with the container size and margin
    Arguments:
        pos1 (Position): The initial position
        pos2 (Position): The destination position
    Returns:
        a tuple with all of the distances (floats)
    """
    extra_height = 2. if (pos1.y == 4) ^ (pos2.y == 4) else 0.
    return (
        abs(pos1.x - pos2.x) * (CONTAINER_LENGTH + MARGIN),
        abs(pos1.y - pos2.y) * (CONTAINER_HEIGHT + extra_height),
        abs(pos1.z - pos2.z) * (CONTAINER_WIDTH + MARGIN),
    )

def calculate_cycle_time(distance: tuple, components=False):
    """
    Calculate movement time in the x, y and z axis for a distance

    Arguments:
        distance (tuple[float]): the distance in the x, y and z direction
        components (bool): if True return the acceleration, constant and 
                           deceleration time as a tuple. Else return the 
                           maximum time

    Returns:
        the cycle time
    """
    a = ACCELERATION
    v = MAX_VELOCITY
    t = []

    for i, x_total in enumerate(distance):
        acceleration_time = v[i] / a[i]
        acceleration_distance = 0.5 * a[i] * acceleration_time ** 2

        if 2 * acceleration_distance >= x_total:
            acceleration_time = sqrt(x_total / a[i])
            if components:
                t.append((acceleration_time, 0.))
            else:
                t.append(acceleration_time)

        constant_speed_distance = x_total - acceleration_distance * 2
        constant_speed_time = constant_speed_distance / v[i]
        if x_total == 0:
            acceleration_time = constant_speed_time = 0
        if components:
            t.append((acceleration_time, constant_speed_time))
        else:
            t.append(acceleration_time * 2 + constant_speed_time)

    if not components:
        return max(t)
    return t


def calculate_work(distance: tuple) -> float:
    """
    Calculate the work for each direction

    Arguments:
        distance (tuple[float]): distance in each direction

    Returns:
        the work for that distance
    """
    times = calculate_cycle_time(distance, components=True)
    a = ACCELERATION
    v = MAX_VELOCITY
    a_res = ACCELERATION_RESISTANCE
    m = MASS
    work = 0
    for i in range(len(distance)):
        work += 0.5 * m * (a_res[i] + a[i]) * a[i] * times[i][0] ** 2
        work += m * a_res[i] * v[i] * times[i][1]
        work += 0.5 * m * (a_res[i] - a[i]) * a[i] * times[i][0] ** 2

    return work / 3_600_000.

def get_free_spot_linear(warehouse, position,
                         truck_position, max_height) -> Position:
    """
    find a free spot in the warehouse to store a container
    Arguments:
        warehouse (tuple[list[int]]): the container layout of the warehouse
        position (Position): The position of the container that has to be freed
        truck_position (Position): The position of the truck
        max_height (int): The maximum height a container can stack on
    Returns:
        A position of a free spot

    Raises
        IndexError when there is no free spot in the warehouse
    """
    _ = position
    for x in range(len(warehouse)):
        for z in range(len(warehouse[0])):
            if warehouse[x][z] < max_height and (x != truck_position.x or
                                                 z != truck_position.z):
                return Position(x, warehouse[x][z], z)
    raise IndexError("no free positions in warehouse for temp storage")

def get_nearest_free_spot(warehouse, position, 
                          truck_position, max_height) -> Position:
    """
    find a free spot in the warehouse to store a container using a radial 
    search
    Arguments:
        warehouse (tuple[list[int]]): the container layout of the warehouse
        position (Position): The position of the container that has to be freed
        truck_position (Position): The position of the truck
        max_height (int): The maximum height a container can stack on
    Returns:
        A position of a free spot

    Raises
        IndexError when there is no free spot in the warehouse
    """
    size = 0
    x_dir = 0
    y_dir = 0
    while True:
        current = Position(position.x - (size + 1), 0, position.z)
        found_candidates = 0
        for i in range(4):
            x_dir = 1 if i < 2 else -1
            y_dir = 1 if i % 3 == 0 else -1
            for _ in range(size + 1):
                current = Position(current.x + x_dir, 0, current.z + y_dir)
                if (current.x < 0 or current.z < 0 or
                    current.x >= len(warehouse) or
                    current.z >= len(warehouse[0]) or
                    (current.x == truck_position.x and
                    current.z == truck_position.z)):
                    continue
                elif warehouse[current.x][current.z] < max_height:
                    current.y = warehouse[current.x][current.z]
                    return current
                found_candidates += 1
        if found_candidates == 0:
            raise IndexError("no free positions in warehouse for temp storage")
        size += 1

def get_full_route(container: Position, truck_position: Position, 
                   warehouse_contents, warehouse_size, retreval_function):
    """
    Get the full path for moving a container to the truck.
    This includes temporary movements of containers that are in the way
    Arguments:
        container (Position): The position of the container which has to be 
                              moved
        truck_position (Position): The position of the truck
        warehouse_contents (tuple[list[int]]): The container layout in 
                                               the warehouse
        warehouse_size (Size): The dimentions of the warehouse
        retreval_function (Callable): The function to use when finding a free 
                                      spot
    Returns:
        A CranePath object with the full path of the crane (including attaching 
        and detaching containers)

    Raises:
        IndexError when invalid coordinates are supplied or when there's no
        free space in the warehouse when its needed
    """

    attach_and_detach_speed = moving_speed = SIMULATION_SPEED

    path: CranePath = CranePath(warehouse_size, moving_speed,
                                attach_and_detach_speed)

    max_height = warehouse_size.y
    current_pos = Position(0,max_height,0)
    path.move_to(current_pos)

    temp_spots = []

    if container.x >= warehouse_size.x or container.z >= warehouse_size.z:
        raise IndexError(f"invalid container has been supplied: {container}")
    elif container.x < 0 or container.z < 0 or container.y < 0:
        raise IndexError("coordinates with negative numbers are not allowed")
    elif container.y >= warehouse_contents[container.x][container.z]:
        raise IndexError(f"invalid container coordinate: {container}")



    while warehouse_contents[container.x][container.z] - 1 > container.y:
        free_spot = retreval_function(warehouse_contents, container, 
                                      truck_position, warehouse_size.y)
        path.move_to(Position(container.x, current_pos.y, container.z))
        current_pos = deepcopy(container)
        current_pos.y = warehouse_contents[container.x][container.z] - 1
        path.move_to(current_pos).attach()
        warehouse_contents[container.x][container.z] -= 1
        current_pos.y = max_height
        path.move_to(current_pos)
        path.move_to(Position(free_spot.x, current_pos.y, free_spot.z))
        current_pos = free_spot
        path.move_to(current_pos).detach()
        temp_spots.append(deepcopy(free_spot))
        warehouse_contents[free_spot.x][free_spot.z] += 1
        current_pos.y = max_height
        path.move_to(current_pos)

    current_pos = deepcopy(container)
    current_pos.y = max_height
    path.move_to(current_pos)
    path.move_to(container)
    path.attach()
    warehouse_contents[container.x][container.z] -= 1
    current_pos.y = max_height
    path.move_to(current_pos)
    truck_position.y = max_height
    current_pos = truck_position
    path.move_to(current_pos)
    current_pos.y = warehouse_contents[current_pos.x][current_pos.z]
    path.move_to(current_pos).detach()
    warehouse_contents[current_pos.x][current_pos.z] += 1
    current_pos.y = max_height
    path.move_to(current_pos)

    for spot in reversed(temp_spots):
        current_pos = deepcopy(spot)
        current_pos.y = max_height
        path.move_to(current_pos).move_to(spot).attach().move_to(current_pos)
        warehouse_contents[spot.x][spot.z] -= 1
        current_pos = deepcopy(container)
        current_pos.y = max_height
        path.move_to(current_pos)
        current_pos.y = warehouse_contents[container.x][container.z]
        path.move_to(current_pos).detach()
        warehouse_contents[container.x][container.z] += 1
        current_pos.y = max_height
        path.move_to(current_pos)

    path.idle(1)
    return path


def read_csv(filename: str, delimiter=';', conversion_type=int, 
             expected_length=3, no_check=False):
    """
    yield rows of a csv file

    Parameters:
        filename (str): the name/path of the csv file
        delimiter (str): the delimiter the csv file should use (default = ';')
        conversion_type (type): the type to convert each 
                                cell to (default = int)
        expected_length (int): The amount of columns in each row (default = 3)
        no_check (bool): Don't check the amount of columns (default = False)
    Returns:
        None

    Raises:
        FileNotFoundError if the file does not exist
    """
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=delimiter)

        for row in reader:
            if len(row) != expected_length and not no_check:
                raise ValueError(f"expected only {expected_length} numbers per"
                                 f" row not {len(row)}!")
            yield [conversion_type(col) for col in row]

def write_csv(filename: str, rows: list[tuple], delimiter=';'):
    """
    write a list of list (rows) to a csv file.
    Parameters:
        filename (str): the name/path of the csv file
        rows: (list[list] | tuple[list]): a list of rows to store in the 
                                          csv file
        delimiter (str): the delimiter the csv file should use (default = ';')

    Returns:
        None

    Raises:
        FileNotFoundError if the file does not exist
    """
    if filename == '-':
        print(*(';'.join((str(col) for col in row)) for row in rows), sep='\n')
        return
    with open(filename, "w") as file:
        for row in rows:
            file.write(';'.join((str(col) for col in row)))
            file.write('\n')

def read_stdin_numbers(delimiter=';'):
    """
    Read numbers in csv format from stdin
    Arguments:
        delimiter: the delimiter between the numbers (default = ';')
    Yields:
        a tuple containing the parsed numbers from a line from stdin
    """
    try:
        while True:
            line = input().split(delimiter)
            if len(line) != 3:
                raise ValueError("stdin should supply 3 numbers as a position "
                                 f"not {len(line)}!")
            yield (int(line[0]), int(line[1]), int(line[2]))
    except EOFError:
        raise StopIteration


def application_loop(warehouse_contents: tuple[list], 
                     warehouse_size: Size, 
                     reset_warehouse: bool,
                     positions: Generator, 
                     truck_position: Position, 
                     retreval_function: Callable,
                     app: Application | None, 
                     debug=True):
    """
    The main application loop
    Arguments:
        warehouse_contents (tuple[list[int]]): The container layout in 
                                               the warehouse
        warehouse_size (Size): The dimentions of the warehouse
        reset_warehouse (bool): Reset the warehouse state after each cycle
        positions (Generator): A generator that yields the next position in the
                               sequence of positions that has been supplied
                               When this argument is None, the positions 
                               will be asked by the display engine (when gui is 
                               enabled)
        truck_position (Position): The position of the truck
        retreval_function (Callable): The function to use when finding a free 
                                      spot
        app (Application | None): The application instance (for the gui 
                                  display). When None don't use a gui.
    Returns:
        A list of cycle time and work values which have been calculated for
        each input
    """
    results = []

    gui = True
    if app is None:
        gui = False

    try:
        while True:
            coords: Position
            if gui and reset_warehouse:
                app.fill_warehouse(
                    *warehouse_contents
                )
                app.set_crane_position(truck_position)

            if gui and positions is None:
                coords= app.get_container()
            else:
                try:
                    coords = Position(*next(positions))
                except StopIteration:
                    break

            contents = warehouse_contents
            if reset_warehouse:
                contents = deepcopy(warehouse_contents)
            path = get_full_route(coords, truck_position, contents, 
                                  warehouse_size, retreval_function)

            work: float = 0.
            time: float = 0.

            initial: Position = truck_position

            for cmd in path._cmds:
                match cmd[0]:
                    case 'M':
                        distance = translate_distances(initial, cmd[3])
                        initial = cmd[3]
                        work += calculate_work(distance)
                        time += calculate_cycle_time(distance)
                    case 'A':
                        time += ATTACH_TIME
                    case 'D':
                        time += DETACH_TIME
                    case _:
                        ...
            if gui:
                app.exec(path)
                app.containers[truck_position.x][truck_position.z] = 0
                app.update_text(f"Last results:\nCycle time: {time / 60:.2f} "
                                f"min\nEnergy cost: {work:.2f} kWh")

            results.append((time, work))
            debug_print(f"{time = } s = {time /60:.1f} min - avg = {sum(t for t, _ in results) / len(results)} ")
            debug_print(f"{work = } kWh")
    except Exception as e:
        raise Exception(f"At input {len(results)}, the following exception "
                        f"occured\n{e}")

    return results


def cli(warehouse_contents: tuple[list], warehouse_size: Size, 
        reset_warehouse: bool,
        positions: Generator, truck_position: Position, 
        retreval_function: Callable):
    """
    run the main application loop without a gui
    Arguments:
        warehouse_contents (tuple[list[int]]): The container layout in 
                                               the warehouse
        warehouse_size (Size): The dimentions of the warehouse
        reset_warehouse (bool): Reset the warehouse state after each cycle
        positions (Generator): A generator that yields the next position in the
                               sequence of positions that has been supplied
                               When this argument is None, the positions 
                               will be asked by the display engine (when gui is 
                               enabled)
        truck_position (Position): The position of the truck
        retreval_function (Callable): The function to use when finding a free 
                                      spot
    """

    return application_loop(warehouse_contents, 
                            warehouse_size, 
                            reset_warehouse,
                            positions, 
                            truck_position, 
                            retreval_function,
                            None)


def gui(warehouse_contents: tuple[list], warehouse_size: Size,
        reset_warehouse: bool,
        positions: Generator, truck_position: Position, 
        retreval_function: Callable):
    """
    run the main application loop with a gui
    Arguments:
        warehouse_contents (tuple[list[int]]): The container layout in 
                                               the warehouse
        warehouse_size (Size): The dimentions of the warehouse
        reset_warehouse (bool): Reset the warehouse state after each cycle
        positions (Generator): A generator that yields the next position in the
                               sequence of positions that has been supplied
                               When this argument is None, the positions 
                               will be asked by the display engine (when gui is 
                               enabled)
        truck_position (Position): The position of the truck
        retreval_function (Callable): The function to use when finding a free 
                                      spot
    """

    with Application(warehouse_size) as app:
        return application_loop(warehouse_contents,
                                warehouse_size,
                                reset_warehouse,
                                positions,
                                truck_position,
                                retreval_function,
                                app)

def debug_print(*args, **kwargs):
    """
    Print to stderr

    Arguments:
        args (tuple[any]): positional arguments
        kwargs (dict[str, any]): keyword arguments
    Returns:
        None
    """
    if DEBUG:
        print(*args, **kwargs, file=stderr)


def main():
    """
    The main function of the program
    Needs further detailing
    """

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="A warehouse crane simulator for determining cycle times "
                    "and energy consumption",
        epilog="Thanks for using main.py!",
        allow_abbrev=False
    )

    warehouse_group = parser.add_mutually_exclusive_group()

    parser.add_argument('-i', '--input-file', 
                        help="input csv file with retreval positions")
    warehouse_group.add_argument('-w', '--warehouse-config-file', 
                        help="input csv file with the warehouse contents")
    parser.add_argument('-o', '--output-file', 
                        help="output csv file with the cycle time and "
                             "work values")
    parser.add_argument('-R', '--no-warehouse-reset',
                       help="Don't reset the warehouse after each retrival",
                       action="store_true")
    parser.add_argument('-A', '--tmp-store-algorithm',
                        help="give the name of a store algorithm")
    parser.add_argument('-S', '--simulation-speed',
                        type=float,
                        help="time of each step in ms")
    parser.add_argument('-t', '--truck-position',
                        nargs=3,
                        type=int,
                        default=Position(0,0,0),
                        help="the position of the truck")
    parser.add_argument('-v', '--speed-vector',
                        nargs=3,
                        type=float,
                        help="the maximum speed in each direction of the "
                             "crane in m/s")
    parser.add_argument('-a', '--acceleration-vector',
                        nargs=3,
                        type=float,
                        help="the acceleration in each direction of the crane "
                             "in m/s^2")
    warehouse_group.add_argument('-d', '--warehouse-dimentions',
                        nargs=3,
                        type=int,
                        help="the dimentions of the warehouse normalized to"
                             " the container dimentions")
    parser.add_argument('-m', '--container-mass',
                        type=int,
                        help="the mass of each container in kg")
    parser.add_argument('-c', '--cli-only',
                       help="Don't use the gui",
                       action="store_true")

    parser.add_argument('--debug',
                       help="print debug output to stderr",
                       action="store_true")

    # parse the arguments
    # errors will be handled by argparse
    args: argparse.Namespace = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    if args.speed_vector is not None:
        global MAX_VELOCITY
        MAX_VELOCITY = args.speed_vector

    if args.acceleration_vector is not None:
        global ACCELERATION
        ACCELERATION = args.acceleration_vector

    if args.container_mass is not None:
        global MASS
        MASS = args.container_mass

    if args.simulation_speed is not None:
        global SIMULATION_SPEED
        SIMULATION_SPEED = args.simulation_speed

    truck_position = args.truck_position

    # get the input file
    input_positions: Generator | None = None
    if args.input_file is not None:
        input_positions = read_csv(args.input_file)
    elif not isatty(stdin.fileno()):
        input_positions = read_stdin_numbers()

    reset_warehouse = True
    if args.no_warehouse_reset:
        reset_warehouse = False
    
    output_file = "-"
    if args.output_file is not None:
        output_file = args.output_file

    retreval_function = get_nearest_free_spot
    if args.tmp_store_algorithm is not None:
        if args.tmp_store_algorithm == "LIN":
            retreval_function = get_free_spot_linear


    # create the warehouse layout
    warehouse_size: Size = Size(8, 5, 10)
    warehouse_contents: tuple
    if args.warehouse_config_file is not None:
        warehouse_contents = tuple(read_csv(args.warehouse_config_file, 
                                            no_check=True))
    else:
        if args.warehouse_dimentions is not None:
            warehouse_size = Size(*args.warehouse_dimentions)
        warehouse_contents = tuple([warehouse_size.y for _ in range(
                                    warehouse_size.z)] for _ in range(
                                    warehouse_size.x))
        warehouse_contents[truck_position.x][truck_position.z] = 0
        if truck_position.x < warehouse_size.x:
            warehouse_contents[truck_position.x + 1][truck_position.z] = 0
        else:
            warehouse_contents[truck_position.x - 1][truck_position.z] = 0

    debug_print("warehouse_config:")
    debug_print(*warehouse_contents, sep='\n')
    debug_print(warehouse_size)

    output: list
    try:
        if args.cli_only:
            output = cli(warehouse_contents, warehouse_size, reset_warehouse, 
                         input_positions, truck_position, retreval_function)
        else:
            output = gui(warehouse_contents, warehouse_size, reset_warehouse, 
                         input_positions, truck_position, retreval_function)
    except Exception as e:
        print(f"\n\x1b[31m{e}\x1b[0m\n\n\tExiting...", file=stderr)
        if DEBUG:
            traceback.print_exc(file=stderr)
        exit(1)

    write_csv(output_file, output)


if __name__ == '__main__':
    main()
