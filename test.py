import unittest
from crane_controller import *
from crane_simulator import *


class Vec3iTest(unittest.TestCase):
    def test_fields(self):
        vec: Vec3i = Vec3i(1602, 6502, 80)
        self.assertEqual(vec.x, 1602)
        self.assertEqual(vec.y, 6502)
        self.assertEqual(vec.z, 80)

    def test_repr(self):
        vec: Vec3i = Vec3i(1, 2, 3)
        self.assertEqual(f"{vec}", "Vec3i[x: 1, y: 2, z: 3]")

    def test_eq(self):
        vec1: Vec3i = Vec3i(1, 1, 1)
        vec2: Vec3i = Vec3i(1, 1, 2)
        self.assertNotEqual(vec1, vec2)
        vec2.z = 1
        self.assertEqual(vec1, vec2)


class SmoothstepTest(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp(-1.0), 0)
        self.assertEqual(clamp(2.0), 1)
        self.assertEqual(clamp(0.5), 0.5)
        self.assertEqual(clamp(-2, lower_limit=-1.0), -1.0)
        self.assertEqual(clamp(3, upper_limit=2.0), 2.0)

    def test_smoothstep(self):
        self.assertEqual(smoothstep(0, 1, 0), 0)
        self.assertEqual(smoothstep(0, 1, 1), 1)
        self.assertEqual(smoothstep(0, 1, 0.5), 0.5)
        # TODO: add more test cases!


class CranePathTest(unittest.TestCase):
    def test_speed(self):
        with self.assertRaises(ValueError):
            _ = CranePath(Size(1, 1, 1), 1001, 1)

        with self.assertRaises(ValueError):
            _ = CranePath(Size(1, 1, 1), 0, 1)

        with self.assertRaises(ValueError):
            _ = CranePath(Size(1, 1, 1), 1, 1001)

        with self.assertRaises(ValueError):
            _ = CranePath(Size(1, 1, 1), 1, 0)

    def test_check_position(self):
        path = CranePath(Size(2, 2, 2), 1, 1)
        with self.assertRaises(ValueError):
            path._check_position(Position(2, 0, 0))

        with self.assertRaises(ValueError):
            path._check_position(Position(0, 3, 0))

        with self.assertRaises(ValueError):
            path._check_position(Position(0, 0, 2))

        with self.assertRaises(ValueError):
            path._check_position(Position(0, 0, -1))

    def test_cmds(self):
        path = CranePath(Size(2, 2, 2), 1, 1)
        path.move_to(Position(0, 0, 0))
        path.attach()
        path.detach()
        path.idle(1000)
        self.assertEqual(path._cmds, [('M', 0, 1000, Position(0, 0, 0)),
                                      ('A', 1000, 2000),
                                      ('D', 2000, 3000),
                                      ('I', 3000, 4000)
                                      ])

    def test_idle(self):
        path = CranePath(Size(2, 2, 2), 1, 1)
        with self.assertRaises(ValueError):
            path.idle(0)

    def test_calculate_duration(self):
        path = CranePath(Size(2, 2, 2), 1, 1)
        self.assertEqual(path._calculate_duration(1000), (0, 1000))

class TranslateDistanceTest(unittest.TestCase):
    def test_output(self):
        pos1: Position = Position(0,0,0)
        pos2: Position = Position(1,0,0)
        self.assertEqual(translate_distances(pos1, pos2), (CONTAINER_LENGTH + MARGIN, 0., 0.))
        pos2 = Position(0,1,0)
        self.assertEqual(translate_distances(pos1, pos2), (0., CONTAINER_HEIGHT, 0.))
        pos2 = Position(0,0,1)
        self.assertEqual(translate_distances(pos1, pos2), (0., 0., CONTAINER_WIDTH + MARGIN))

class CalculateCycleTime(unittest.TestCase):
    def test_output(self):
        distance: tuple = (0, 10, 0)
        self.assertEqual(calculate_cycle_time(distance), 75.38095238095238)
        distance = (10, 0, 0)
        self.assertEqual(calculate_cycle_time(distance), 18.333333333333336)
        distance = (0, 0, 10)
        self.assertEqual(calculate_cycle_time(distance), 21.416666666666664)

    def test_single_output(self):
        distance = (10, 0, 0)
        self.assertEqual(type(calculate_cycle_time(distance)), float)

    def test_multiple_output(self):
        distance = (10, 0, 0)
        out = calculate_cycle_time(distance, components=True)
        self.assertEqual(type(out), list)
        self.assertEqual(type(out[0]), tuple)
        self.assertEqual(len(out), 3)

class CalculateWork(unittest.TestCase):
    def test_output(self):
        distance: tuple = (0, 10, 0)
        self.assertEqual(calculate_work(distance), 0.872)

class GetFreeSpotLinear(unittest.TestCase):
    def test_output(self):
        warehouse: tuple = (
            [0, 2, 1],
            [2, 1, 2],
            [2, 2, 2],
        )
        position: Position = Position(2, 1, 1)
        truck: Position = Position(0, 0, 0)
        max_height: int = 2
        spot: Position = get_free_spot_linear(warehouse, position, 
                                              truck, max_height)

        self.assertEqual(spot, Position(0, 1, 2))

    def test_no_free_spots(self):
        warehouse: tuple = (
            [0, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        )
        position: Position = Position(2, 1, 1)
        truck: Position = Position(0, 0, 0)
        max_height: int = 2
        with self.assertRaises(IndexError):
            _ = get_free_spot_linear(warehouse, position,
                                     truck, max_height)


class GetNearestFreeSpot(unittest.TestCase):
    def test_output(self):
        warehouse: tuple = (
            [0, 2, 1],
            [2, 1, 2],
            [2, 2, 2],
        )
        position: Position = Position(2, 1, 1)
        truck: Position = Position(0, 0, 0)
        max_height: int = 2
        spot: Position = get_nearest_free_spot(warehouse, position,
                                               truck, max_height)
        self.assertEqual(spot, Position(1, 1, 1))

    def test_no_free_spots(self):
        warehouse: tuple = (
            [0, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        )
        position: Position = Position(2, 1, 1)
        truck: Position = Position(0, 0, 0)
        max_height: int = 2
        with self.assertRaises(IndexError):
            _ = get_nearest_free_spot(warehouse, position,
                                      truck, max_height)

class GetFullRoute(unittest.TestCase):
    def test_output(self):
        warehouse_size = Size(3, 2, 3)
        warehouse: tuple = (
            [0, 2, 2],
            [2, 1, 2],
            [2, 2, 2],
        )
        truck: Position = Position(0, 0, 0)
        container = Position(2, 0, 1)
        path = get_full_route(container, truck, warehouse,
                              warehouse_size, get_nearest_free_spot)

        move_speed = attach_speed = 10
        path_cmp = CranePath(warehouse_size, move_speed, attach_speed)
        (
            path_cmp.move_to(Position(0, 2, 0))
                    .move_to(Position(2, 2, 1))
                    .move_to(Position(2, 1, 1))
                    .attach()
                    .move_to(Position(2, 2, 1))
                    .move_to(Position(1, 2, 1))
                    .move_to(Position(1, 1, 1))
                    .detach()
                    .move_to(Position(1, 2, 1))
                    .move_to(Position(2, 2, 1))
                    .move_to(Position(2, 0, 1))
                    .attach()
                    .move_to(Position(2, 2, 1))
                    .move_to(Position(0, 2, 0))
                    .move_to(Position(0, 0, 0))
                    .detach()
                    .move_to(Position(0, 2, 0))
                    .move_to(Position(1, 2, 1))
                    .move_to(Position(1, 1, 1))
                    .attach()
                    .move_to(Position(1, 2, 1))
                    .move_to(Position(2, 2, 1))
                    .move_to(Position(2, 0, 1))
                    .detach()
                    .move_to(Position(2, 2, 1))
                    .idle(1)
        )
        self.assertEqual(path._cmds, path_cmp._cmds)

    def test_invalid_position(self):
        warehouse_size = Size(3, 2, 3)
        warehouse: tuple = (
            [0, 2, 2],
            [2, 1, 2],
            [2, 2, 2],
        )
        truck: Position = Position(0, 0, 0)
        container = Position(3, 1, 2)

        with self.assertRaises(IndexError):
            _ = get_full_route(container, truck, warehouse,
                               warehouse_size, get_nearest_free_spot)


        container = Position(-2, 1, 2)

        with self.assertRaises(IndexError):
            _ = get_full_route(container, truck, warehouse,
                               warehouse_size, get_nearest_free_spot)

        container = Position(2, 2, 2)


        with self.assertRaises(IndexError):
            _ = get_full_route(container, truck, warehouse,
                               warehouse_size, get_nearest_free_spot)


    def test_no_free_space(self):
        warehouse_size = Size(3, 2, 3)
        warehouse: tuple = (
            [0, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        )
        truck: Position = Position(0, 0, 0)
        container = Position(1, 0, 2)

        with self.assertRaises(IndexError):
            _ = get_full_route(container, truck, warehouse,
                               warehouse_size, get_nearest_free_spot)


class ReadCsv(unittest.TestCase):
    def test_output(self):
        filename: str = "_tmp_test_file_code_bram.csv"
        with open(filename, "w") as fd:
            fd.write("1;2;3\n4;5;6\n")

        result = tuple(read_csv(filename, delimiter=';', conversion_type=int))
        self.assertEqual(result, ([1,2,3], [4,5,6]))

        from os import remove
        remove(filename)

    def test_invalid_file_name(self):
        with self.assertRaises(FileNotFoundError):
            tuple(read_csv("__this_really_long_file_name_does_hopefully_not_"
                           "exist.csv", 
                           delimiter=';', conversion_type=int))

    def test_invalid_input(self):
        filename: str = "_tmp_test_file_code_bram.csv"
        with open(filename, "w") as fd:
            fd.write("1;2\n4;5\n")

        with self.assertRaises(ValueError):
            _ = tuple(read_csv(filename, delimiter=';', 
                               conversion_type=int))

        from os import remove
        remove(filename)

class WriteCsv(unittest.TestCase):
    #def write_csv(filename: str, rows: list[list], delimiter=';'):
    def test_output(self):
        filename: str = "_tmp_test_file_code_bram.csv"
        rows = [(1,2), (1,2)]
        write_csv(filename, rows, delimiter=';')
        with open(filename) as fd:
            filecontents = fd.read()
            self.assertEqual(filecontents, "1;2\n1;2\n")

        from os import remove
        remove(filename)


class ReadStdinNumber(unittest.TestCase):
    # see ReadCsv test
    ...

class ApplicationLoop(unittest.TestCase):
    def test_output(self):
        warehouse_size = Size(3, 2, 3)
        warehouse: tuple = (
            [0, 2, 2],
            [2, 1, 2],
            [2, 2, 2],
        )
        truck: Position = Position(0, 0, 0)

        filename: str = "_tmp_test_file_code_bram.csv"
        with open(filename, "w") as fd:
            fd.write("2;1;2\n2;0;1\n")

        positions = read_csv(filename, delimiter=';', conversion_type=int)

        
        out = application_loop(warehouse, warehouse_size, True, positions,
                               truck, get_nearest_free_spot, None, False)

        cmp = [(182.63347619047622, 1.6754689777777778),
               (510.05242857142855, 4.587082488888889)]

        from os import remove
        remove(filename)

        self.assertEqual(out, cmp)

    # def test_raises(self):

