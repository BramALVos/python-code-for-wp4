R"""
This is a modified version of the original crane_controller.py file
from the repo: https://github.com/BramALVos/crane_controller

All modifications where made by Bram Vos (original developer of the module)
without the help of AI

originally deveolped by:
Copyright (c) 2025 Bram Vos (vos0127@hz.nl)
Copyright (c) 2025 gwaadiegwaa
"""

import threading
import time
from typing import Self
from copy import deepcopy
from enum import Enum

try:
    import pyray as rl
except ImportError as e:
    print(e)
    print("Please install `raylib` using pip! (`pip install raylib`)")
    exit(1)


class Vec3i:
    """
    3D Integer vector class
    This class serves as the baseclass for the Position and Size class
    """

    def __init__(self, x: int, y: int, z: int):
        """
        Initialize a Vec3i
        Parameters:
            x (int): The x value of the vector
            y (int): The y value of the vector
            z (int): The z value of the vector
        Returns:
            A new Vec3i with the passed values for x, y and z
        """
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def __repr__(self):
        """
        Return a string representation of a Vec3i or a child of Vec3i
        """
        return (f"{self.__class__.__name__}"
                f"[x: {self.x}, y: {self.y}, z: {self.z}]")

    def __eq__(self, other):
        """
        Compare two Vec3i's
        Parameters:
            other (Vec3i): The Vec3i to compare to self
        Returns:
            True when self is equal to other, else return False
        """
        return self.x == other.x and self.y == other.y and self.z == other.z


class Position(Vec3i):
    """
    A position in the 3D space of the warehouse.
    Parameters:
        x (int): The x value of the vector
        height (int): The y value of the vector
        z (int): The z value of the vector
    Returns:
        A new Vec3i with the passed values for x, y and z
    """

    def __init__(self, x: int, height: int, z: int):
        super().__init__(x, height, z)


class Size(Vec3i):
    """
    A size in 3D space.
    Parameters:
        width (int): The x value of the vector
        height (int): The y value of the vector
        length (int): The z value of the vector
    Returns:
        A new Vec3i with the passed values for x, y and z
    """

    def __init__(self, width: int, height: int, length: int):
        super().__init__(width, height, length)


def clamp(x: float, lower_limit: float = 0.0, upper_limit: float = 1.0):
    """
    clamp a value between two points
    Parameters:
        x (float): The value to be clamped
        lower_limit (float): The lower_limit of the clamp function 
                             (aka the lowest possible value)
                             Default value = 0.0

        upper_limit (float): The upper_limit of the clamp function 
                             (aka the biggest possible value)
                             Default value = 1.0

    Returns:
        x or lower_limit when x < lower_limit or upper_limit
        when x > upper_limit
    """
    return min(max(x, lower_limit), upper_limit)


def smoothstep(edge0: float, edge1: float, x: float):
    """
    Return a value between 0 and 1 with a smooth transition at the beginning
    and end.

    Adapted from: https://en.wikipedia.org/wiki/Smoothstep
    Parameters:
        edge0 (float): The lowest value that x can be
        edge1 (float): The biggest value that x can be
        x (float): The value to be used in the smoothstep function

    Returns:
        3x^2 - 2x^3 where x has been clamped with the help of edge0 and edge1
    """
    x = clamp((x - edge0) / (edge1 - edge0))
    return (3 - 2 * x) * (x ** 2)  # 3x^2 - 2x^3


def lerp(first, second, time):
    """
    Calculate the position between two positions given a time
    see wikipedia with searchterm Lerp

    Arguments:
        first (float): first position
        second (float) second position
        time (float between 0 and 1): the time index for the lerp
    Returns:
        float
    """
    return first + time * (second - first)


class CranePath:
    """
    This class is used for chaining crane commands.
    They can be passed to Application.exec() to execute them
    """

    def __init__(self, warehouse_size: Size, move_speed: float,
                 attach_detach_speed: float):
        """
        Initialize a CranePath structure.
        Commands can be pushed to this structure later on.
        """
        if not (0. < move_speed < 1000.):
            raise ValueError("move_speed must be between 0 and 1000")
        elif not (0. < attach_detach_speed < 1000.):
            raise ValueError("attach_detach_speed must be between 0 and 1000")

        self._cmds: list[tuple] = []
        self._move_speed = int(1000. / move_speed)
        self._attach_detach_speed = int(1000. / attach_detach_speed)
        self._warehouse_size = deepcopy(warehouse_size)
        self._warehouse_size.y += 1

    def __len__(self):
        """
        Return the command count
        """
        return len(self._cmds)

    def _check_position(self, position: Position) -> None:
        """
        Determine if a coordinate is valid (aka does it fit in the warehouse)
        Parameters:
            position (Position): The coordinate to be checked
        Returns:
            None
        Raises:
            a ValueError when the coordinate is not inside the warehouse
        """
        if position.x < 0 or position.y < 0 or position.z < 0:
            raise ValueError("x, y, or z may not be less than 0\n"
                             f"{position}")
        if position.x >= self._warehouse_size.x:
            raise ValueError("invalid x dimension "
                             f"(max is {self._warehouse_size.x - 1})")
        if position.y >= self._warehouse_size.y:
            raise ValueError("invalid y dimension "
                             f"(max is {self._warehouse_size.y - 2})")
        if position.z >= self._warehouse_size.z:
            raise ValueError("invalid z dimension "
                             f"(max is {self._warehouse_size.z - 1})")

    def attach(self) -> Self:
        """
        Append an attach command. This command wil attach a container 
        to the crane when possible
        """
        self._cmds.append((
            'A', *self._calculate_duration(self._attach_detach_speed)
        ))
        return self

    def detach(self) -> Self:
        """
        Append a detach command. This command wil detach a container 
        from the crane when possible
        """
        self._cmds.append((
            'D', *self._calculate_duration(self._attach_detach_speed)
        ))
        return self

    def move_to(self, position: Position) -> Self:
        """
        Append a move command. This command will move the crane to 
        certain position.
        Parameters:
            position (Position): The position to which the crane should move
        Raises:
            A ValueError when an invalid position is given.
        """
        self._check_position(position)
        position = Position(position.x, position.y, position.z)

        self._cmds.append((
            'M', *self._calculate_duration(self._move_speed), position
        ))
        return self

    def idle(self, duration: int) -> Self:
        """
        Append an idle command. This command will make the crane wait for a 
        duration in ms.
        Parameters:
            duration (int): A duration to wait for in ms
        """
        if duration < 1:
            raise ValueError("duration must be 1 ms or higher")

        self._cmds.append((
            "I", *self._calculate_duration(duration)
        ))
        return self

    def _calculate_duration(self, duration: int):
        """
        Calculate the start and end time for a command.
        Parameters:
            duration (int): The duration of the command
        Returns:
            A tuple with the start and end time of the command
        """
        t_start: int = 0
        if len(self._cmds) > 0:
            t_start = self._cmds[-1][2]

        t_end: int = t_start + duration
        return t_start, t_end

    def __repr__(self):
        """
        Return the string representation of a CranePath.
        """
        result = []
        mapping: dict[str, str] = {'M': "MOVE", 'D': "DETACH",
                                   'A': "ATTACH", 'I': "IDLE"}
        for cmd in self._cmds:
            result.append(f"{mapping[cmd[0]]}")

            if cmd[0] == 'M':
                result.append(f" {cmd[3]}")

            elif cmd[0] == 'I':
                result.append(f" {cmd[2] - cmd[1]}")

            result.append(f" @ {cmd[1]}\n")
        return "".join(result)


def rl_camera_copy(camera: rl.Camera3D):
    """
    Copy a raylib camera (because deepcopy doesn't work for some reason and I 
    am to lazy to figure out how to make that work :))

    Arguments:
        camera (Camera3D): The camera to copy

    Returns:
        a copy of camera
    """
    return rl.Camera3D(rl.Vector3(camera.position.x,
                                  camera.position.y,
                                  camera.position.z),
                       rl.Vector3(camera.target.x,
                                  camera.target.y,
                                  camera.target.z),
                       rl.Vector3(camera.up.x,
                                  camera.up.y,
                                  camera.up.z),
                       camera.fovy,
                       camera.projection)

def rl_camera_lerp(original: rl.Camera3D, 
                   new: rl.Camera3D, time: float) -> rl.Camera3D:
    """
    Lerp the camera from an original position to a new position
    Arguments:
        original (Camera3D): The original camera
        new (Camera3D): The new camera
        time (float): time index for the lerp function

    Returns:
        a new camera object between original and new
    """
    camera = rl.Camera3D()
    camera.fovy = original.fovy
    camera.projection = original.projection
    camera.target = rl.Vector3(lerp(original.target.x, 
                                    new.target.x, 
                                    time),
                               lerp(original.target.y, 
                                    new.target.y, 
                                    time),
                               lerp(original.target.z, 
                                    new.target.z, 
                                    time))

    camera.position = rl.Vector3(lerp(original.position.x, 
                                      new.position.x, 
                                      time),
                                 lerp(original.position.y, 
                                      new.position.y, 
                                      time),
                                 lerp(original.position.z, 
                                      new.position.z, 
                                      time))

    camera.up = rl.Vector3(lerp(original.up.x, 
                                new.up.x, 
                                time),
                           lerp(original.up.y, 
                                new.up.y, 
                                time),
                           lerp(original.up.z, 
                                new.up.z, 
                                time))
    return camera

class ApplicationState(Enum):
    """
    An Enum with the different application states
    """
    GET_CONTAINER_LOCATION = 0,
    RUN_MOVEMENT_SIMULATION = 1
    ASK_SAVE = 2,


class Application:
    """
    This class is responsible for setting up / running a simulation
    It spawns a render thread and is controlled by the main thread.
    This makes it possible to keep the window alive while doing other stuff.
    """

    def __init__(self, warehouse_size: Size, window_width=1280,
                 window_height=720, resizeable=False):
        """
        Initialize controller and spawn the render thread
        Parameters:
            warehouse_size (Size): The size of the warehouse (how many blocks 
                                   should fit in the x, y and z directions)
            window_width (int): The width of the window (default 1280 px)
            window_height (int): The height of the window (default 720 px)
        Returns:
            A ready to use Application (yay!)
        """
        self.window_width = window_width
        self.window_height = window_height
        self.resizeable = resizeable

        self.cmd_lock = threading.Lock()
        self.cmd_list: list[tuple] = []
        self.mode_lock = threading.Lock()
        self.mode = ApplicationState.GET_CONTAINER_LOCATION
        self.plane = deepcopy(warehouse_size)
        self.plane.x += 1
        self.plane.y += 2
        self.containers: list[list[int]] = []
        self.attached_container = False

        self.input_location = Position(0,0,0)

        self._display_text: str = ""

        self._engine_shutdown = False
        self._active_engine_request = threading.Event()
        self._active_engine_request.clear()
        self._start_time: int = 0
        self._engine_thread = threading.Thread(target=Application._engine_run,
                                               args=(self,),
                                               kwargs={})

    def __enter__(self):
        """
        This function forces the user to use Application with a with 
        statement (not really, but it makes it harder). This is needed since 
        raylib (responsible for all the graphics) needs to be deinitialized 
        when the window closes.
        A with statement makes sure that __exit__ will get called and __exit__ 
        will close the window respectfully
        """
        self._engine_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        This function will get called when we exit the with statement which 
        houses the Application (see __enter__ for more information)
        """
        self._engine_shutdown = True
        if self._engine_is_running:
            self._engine_thread.join()

    def _handle_simulation(self):
        """
        Handle the rendering part given a path by exec
        """
        self.current_pos = rl.Vector3(self.crane_starting_pos.x,
                                      self.crane_starting_pos.y,
                                      self.crane_starting_pos.z)
        if not self._active_engine_request.is_set():

            with self.cmd_lock:
                current_cmd = (self.cmd_list[0] if len(self.cmd_list) > 0
                                                else None)

                if current_cmd is not None:
                    time_index = ((time.time_ns() // 1000_000) - 
                                   self._start_time)
                    if time_index >= current_cmd[2]:
                        self._exec_till_cmd_index(time_index)
                        # self.cmd_list.pop(0)
                        if len(self.cmd_list) == 0:
                            self._active_engine_request.set()
                            return

                        current_cmd = self.cmd_list[0]

                        # current_pos gets updated to prevent flickering
                    self.current_pos = rl.Vector3(self.crane_starting_pos.x,
                                                  self.crane_starting_pos.y,
                                                  self.crane_starting_pos.z)
                    match current_cmd[0]:
                        case 'M':
                            i = smoothstep(current_cmd[1],
                                           current_cmd[2],
                                           time_index)

                            self.current_pos.x += (current_cmd[3].x -
                                self.crane_starting_pos.x) * i
                            self.current_pos.y += (current_cmd[3].y -
                                self.crane_starting_pos.y) * i
                            self.current_pos.z += (current_cmd[3].z -
                                self.crane_starting_pos.z) * i
                        case _:
                            ...
                else:
                    self._active_engine_request.set()


    def _handle_get_container(self, camera) -> rl.Camera3D:
        """
        get a container position using interactive user input
        Arguments:
            camera (Camera3D): The original camera
        Returns:
            the new camera
        """
        if self._active_engine_request.is_set():
            return camera
        current_looking_ray = rl.get_screen_to_world_ray(rl.
                                                         get_mouse_position(),
                                                         camera)
        self.selection_size = rl.Vector3(0,0,0)
        self.selection_origin = rl.Vector3(0,0,0)
        if self.mouse_free and not self.layer_selected:
            if self.transition_animation < 1:
                self.transition_animation += rl.get_frame_time()
                self.transition_animation = min(1, self.transition_animation)
                camera = rl_camera_lerp(self.new_camera, self.original_camera, 
                                        self.transition_animation)
                if self.transition_animation == 1:
                    self._active_engine_request.set()
            else:
                for layer in range(self.plane.y - 2, 0, -1):
                    selection_start = rl.Vector3(0.5, layer - 1, 0)
                    selection_end = rl.Vector3(self.plane.x - 0.5, layer, 
                                               self.plane.z)
                    boundingbox = rl.BoundingBox(selection_start,
                                                 selection_end)
                    if rl.get_ray_collision_box(current_looking_ray,
                                                boundingbox).hit:
                        self.selection_size = rl.Vector3(self.plane.x - 1 +
                                                         0.1, 1.1, 
                                                         self.plane.z + 0.1)
                        self.selection_origin = rl.Vector3(self.plane.x / 2,
                                                           layer - 0.5, 
                                                           self.plane.z / 2)
                        if rl.is_mouse_button_pressed(rl.
                                                      MouseButton.
                                                      MOUSE_BUTTON_LEFT):
                            self.layer_selected = True
                            self.active_layer = layer - 1
                            self.transition_animation = 0
                            print(f"{self.layer_selected = }")
                        return camera
                else:
                    self.active_layer = None
        elif self.mouse_free and self.layer_selected:
            if self.transition_animation < 1:
                if self.transition_animation == 0:
                    self.original_camera = rl_camera_copy(camera)
                self.new_camera = rl.Camera3D(
                        rl.Vector3(self.plane.x / 2, self.plane.y + 10, 
                                   self.plane.z / 2),
                        rl.Vector3(self.plane.x / 2, self.plane.y, 
                                   self.plane.z / 2),
                        rl.Vector3(0,0,0),
                        camera.fovy,
                        camera.projection)
                if (self.original_camera.position.x > 
                    self.original_camera.position.z):
                    self.new_camera.up= rl.Vector3(-1 if 
                                                   self.original_camera.
                                                   position.x > 0 else 1,0,0)
                else:
                    self.new_camera.up = rl.Vector3(0,0,-1 
                                                    if self.original_camera.
                                                    position.z > 0 else 1)
                self.transition_animation += rl.get_frame_time()
                self.transition_animation = min(1, self.transition_animation)
                camera = rl_camera_lerp(self.original_camera, self.new_camera, 
                                        self.transition_animation)
            else:
                for x in range(self.plane.x - 1):
                    for z in range(self.plane.z):
                        if self.containers[x][z] <= self.active_layer:
                            continue
                        selection_start = rl.Vector3(0.5 + x,
                                                     self.active_layer, z)
                        selection_end = rl.Vector3(x + 1.5,
                                                   self.active_layer + 1, 
                                                   z + 1)
                        boundingbox = rl.BoundingBox(selection_start,
                                                     selection_end)
                        if rl.get_ray_collision_box(current_looking_ray,
                                                    boundingbox).hit:
                            self.selection_size = rl.Vector3(1.1, 1.1, 1.1)
                            self.selection_origin = rl.Vector3(x + 1,
                                                               self.
                                                               active_layer + 
                                                               0.5, z + 0.5)
                            if rl.is_mouse_button_pressed(rl.
                                                          MouseButton.
                                                          MOUSE_BUTTON_LEFT):
                                self.layer_selected = False
                                self.input_location = Position(x, 
                                                               self.
                                                               active_layer, z)
                                self.transition_animation = 0
                                self.active_layer = None
                            return camera
        return camera

    def set_crane_position(self, position: Position):
        """
        Set the initial crane position to `position`
        """
        with self.mode_lock:
            self.current_pos = deepcopy(position)
            self.crane_starting_pos = deepcopy(position)

    def _engine_run(self):
        """
        This method is responsible for all the rendering and is run as a 
        separate thread.
        It displays the graphics and will execute (through helper functions) 
        the commands or display them (in the case of MOVE)
        """
        self._engine_is_running = True
        rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
        rl.init_window(self.window_width, self.window_height, "Application")
        rl.set_target_fps(60)
        if self.resizeable:
            print("config resisable")
            rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)

        rl.set_exit_key(0)

        camera = rl.Camera3D()
        camera.position = rl.Vector3(20., 10., 0)
        camera.target = rl.Vector3(self.plane.x / 2, self.plane.y / 2, 
                                   self.plane.z / 2)
        camera.up = rl.Vector3(0, 1., 0)
        camera.fovy = 45.0
        camera.projection = rl.CameraProjection.CAMERA_PERSPECTIVE

        self.pole_size: float = 0.25
        self.pole_distance_multiplier: float = 0.5

        self.crane_starting_pos = Position(0, 0, 0)

        self.current_pos = rl.Vector3(self.crane_starting_pos.x,
                                      self.crane_starting_pos.y,
                                      self.crane_starting_pos.z)

        self.mouse_free = False
        self.layer_selected = False
        self.active_layer = None

        self.transition_animation = 1
        self.original_camera = rl_camera_copy(camera)
        self.new_camera = rl_camera_copy(camera)


        self.selection_size = rl.Vector3(0, 0, 0)
        self.selection_origin = rl.Vector3(0, 0, 0)

        rl.disable_cursor()

        while not rl.window_should_close() and not self._engine_shutdown:
            if rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE):
                if self.mouse_free:
                    rl.disable_cursor()
                    self.mouse_free = False
                else:
                    rl.enable_cursor()
                    self.mouse_free = True

            if not self.mouse_free:
                rl.update_camera(camera, rl.CameraMode.CAMERA_THIRD_PERSON)

            display_text = 0
            with self.mode_lock:
                match self.mode:
                    case ApplicationState.RUN_MOVEMENT_SIMULATION:
                        self._handle_simulation()
                    case ApplicationState.GET_CONTAINER_LOCATION:
                        camera = self._handle_get_container(camera)

                display_text = self._display_text

            rl.begin_drawing()
            rl.clear_background(rl.WHITE)
            rl.begin_mode_3d(camera)

            self._draw_crane(self.current_pos)
            self._draw_containers(self.active_layer)

            rl.draw_cube_v(self.selection_origin, self.selection_size, 
                           rl.Color(255, 0, 0, 128))

            # rl.draw_grid(10, 1.)

            rl.end_mode_3d()
            rl.draw_text(f"FPS: {rl.get_fps()}", 10, 10, 20, rl.GRAY)
            rl.draw_text(f"mode: {'select' if self.mouse_free else 'move'}", 
                         10, 60, 30, rl.GREEN if self.mouse_free else rl.RED)
            rl.draw_text("(press ESC to change)", 10, 90, 15, rl.GRAY)
            rl.draw_text(display_text, 10, 120, 20, rl.GRAY)
            rl.end_drawing()

        rl.close_window()
        self._active_engine_request.set()
        self._engine_is_running = False

    def _exec_till_cmd_index(self, t: int):
        """
        Execute commands till a certain time
        This function is needed when commands need to be executed faster than
        the frame rate. This function will execute the commands which have been
        missed almost instantly.
        Parameters:
            t (int): time in ms since exec has been called
        Returns:
            None
        """
        index = self._find_cmd_index(t)
        self._exec_cmd_range(index if index > 0 else 1)

    def _find_cmd_index(self, index: int):
        """
        Figure out which command has to be executed currently
        This is basically binary search with a fancy name :)
        Parameters:
            index (int): a time index into the array of commands in ms
        Returns:
            The index of the current command as an int
        """
        l = 0
        r = len(self.cmd_list) - 1
        while l <= r:
            m = (r + l) // 2
            if self.cmd_list[m][1] == index:
                return m

            if self.cmd_list[m][1] < index:
                l = m + 1
            elif self.cmd_list[m][1] > index:
                r = m - 1

        return (r + l) // 2

    def _exec_cmd_range(self, end: int):
        """
        Execute the first `end` commands in the command list
        Parameters:
            end (int): index to the last command to be executed
        Returns:
            None
        Effects:
            The active simulation status.
            When a container can not be attached or detached it will halt the
            simulation
        """
        for _ in range(end):
            match self.cmd_list[0][0]:
                case 'M':
                    self.crane_starting_pos = self.cmd_list[0][3]
                case 'A':
                    if not self._attach_container(self.crane_starting_pos):
                        self._active_engine_request.set()
                        print("Failed to attach container!")
                case 'D':
                    if not self._detach_container(self.crane_starting_pos):
                        self._active_engine_request.set()
                        print("Failed to detach container!")
                case 'I':
                    ...

            self.cmd_list.pop(0)

    def _draw_containers(self, active_layer=None):
        """
        Draw the containers in the warehouse as a cube
        """
        for ix, z in enumerate(self.containers):
            for iz, x in enumerate(z):
                for y in range(x):
                    pos = rl.Vector3(ix + 1, y + 0.5, iz + 0.5)
                    size = rl.Vector3(1, 1, 1)
                    if active_layer is None or active_layer == y:
                        rl.draw_cube_v(pos, size, rl.Color(0, 121, 241, 255)) 
                        rl.draw_cube_wires_v(pos, size, 
                                             rl.Color(102, 191, 255, 255))


    def _draw_crane(self, position: rl.Vector3):
        """
        Draw the crane at a certain position in the warehouse
        Parameters:
            position (rl.Vector3): The position of the crane
        Returns:
            None
        """
        self._draw_crane_frame(position)
        self._draw_crane_top(position)
        self._draw_crane_hook(position)
        self._draw_crane_container(position)

    def _draw_crane_frame(self, position: rl.Vector3):
        """
        Draw the frame of the crane at a certain position in the warehouse
        Parameters:
            position (rl.Vector3): The position of the crane (which effects 
            the positioning of the frame
        Returns:
            None
        """
        for x in range(0, 2):
            for z in range(-1, 2, 2):
                pos = rl.Vector3(
                    x * self.plane.x,
                    self.plane.y / 2,
                    position.z + z * self.pole_distance_multiplier + 0.5,
                )
                size = rl.Vector3(self.pole_size, self.plane.y, self.pole_size)
                rl.draw_cube_v(pos, size, rl.YELLOW)

                pos.x = 0.5 * self.plane.x
                pos.y = self.plane.y + self.pole_size / 2
                size.x = self.plane.x + self.pole_size
                size.y = self.pole_size
                rl.draw_cube_v(pos, size, rl.ORANGE)

    def _draw_crane_top(self, position: rl.Vector3):
        """
        Draw the top of the crane at a certain position in the warehouse
        Parameters:
            position (rl.Vector3): The position of the crane (which effects 
            the positioning of the top of the crane
        Returns:
            None
        """
        pos = rl.Vector3(
            position.x + 1,
            self.plane.y,
            position.z + 0.5
        )
        size = rl.Vector3(1.5, 0.5, 1 - self.pole_size)
        rl.draw_cube_v(pos, size, rl.RED)

    def _draw_crane_hook(self, position: rl.Vector3):
        """
        Draw the hook and rope of the crane at a certain position in
        the warehouse
        Parameters:
            position (rl.Vector3): The position of the crane (which effects 
            the positioning of the hook and the rope of the crane
        Returns:
            None
        """
        pos = rl.Vector3(
            position.x + 1,
            position.y + 1.1,
            position.z + 0.5
        )
        size = rl.Vector3(0.25, 0.2, 0.25)
        rl.draw_cube_v(pos, size, rl.RED)
        size.y = (self.plane.y - pos.y)
        pos.y = size.y / 2 + pos.y
        size.x = size.z = 0.05
        rl.draw_cube_v(pos, size, rl.RED)

    def _draw_crane_container(self, position: rl.Vector3):
        """
        Draw the container hanging on the hook of the crane at a position when 
        there is a hanging container.
        Parameters:
            position (rl.Vector3): The position of the crane (which effects 
            the positioning of the hanging container (when there is one)
        Returns:
            None
        """
        if self.attached_container:
            pos = rl.Vector3(
                position.x + 1,
                position.y + 0.5,
                position.z + 0.5
            )
            size = rl.Vector3(1, 1, 1)
            rl.draw_cube_v(pos, size, rl.BLUE)
            rl.draw_cube_wires_v(pos, size, rl.RED)

    def _detach_container(self, pos: Position) -> bool:
        """
        Try to detach a container a when the hook is a certain position
        Parameters:
            pos (Position): The position of the crane hook
        Returns:
            True when a container could be detached else False
        """
        if self.containers[pos.x][pos.z] != pos.y:
            return False

        self.containers[pos.x][pos.z] += 1
        self.attached_container = False
        return True

    def _attach_container(self, pos: Position) -> bool:
        """
        Try to attach a container a when the hook is a certain position
        Parameters:
            pos (Position): The position of the crane hook
        Returns:
            True when a container could be attached else False
        """
        if self.containers[pos.x][pos.z] - 1 != pos.y:
            return False

        self.containers[pos.x][pos.z] -= 1
        self.attached_container = True
        return True

    def exec(self, path: CranePath):
        """
        Execute a list of CraneCmd's supplied to append_cmds
        Raises:
            A ThreadError when the render thread (aka engine thread) has died
        """
        with self.mode_lock:
            self.mode = ApplicationState.RUN_MOVEMENT_SIMULATION
        self._start_time = time.time_ns() // 1000_000
        with self.cmd_lock:
            self.cmd_list = path._cmds
        self._active_engine_request.clear()
        self._active_engine_request.wait()
        if not self._engine_is_running:
            raise threading.ThreadError("Engine thread stopped")

    def update_text(self, text: str):
        with self.cmd_lock:
            self._display_text = text
        if not self._engine_is_running:
            raise threading.ThreadError("Engine thread stopped")


    def get_container(self):
        """
        Get a container position from the user
        """
        with self.mode_lock:
            self.mode = ApplicationState.GET_CONTAINER_LOCATION
        self._active_engine_request.clear()
        self._active_engine_request.wait()
        if not self._engine_is_running:
            raise threading.ThreadError("Engine thread stopped")

        return self.input_location

    def fill_warehouse(self, *args):
        """
        Fill the warehouse with containers
        Parameters:
        args (tuple[list[int]]): A tuple where each element is a list which
                                 contains integers. The integers represent the 
                                 height/y coordinate (aka how many boxes are
                                 stacked).
                                 For each z coordinate there should be a list,
                                 meaning that the entries in a list represent
                                 the x coordinate.
                                 See the README for more information

        Returns:
            None
        Raises:
            A ValueError when x, y, or z is not inside the warehouse or a
            TypeError when args contains a type other than list
        """
        # we have to do this to make sure that the internal array doesn't
        # change before exec is called
        args = deepcopy(args)
        self.containers.clear()
        if len(args) >= self.plane.x:
            raise ValueError("invalid x dimension (max x = "
                             f"{self.plane.x - 1})")

        for arg in args:
            if type(arg) != list:
                raise TypeError("not a list!")
            elif len(arg) > self.plane.z:
                raise ValueError("invalid z dimension (max z = "
                                 f"{self.plane.z})")
            for a in arg:
                if a >= self.plane.y - 1:
                    raise ValueError("invalid y dimension (max y = "
                                     f"{self.plane.y - 2})")
            self.containers.append(arg)
