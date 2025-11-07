# Guide for using this software

### Installation
1. Donwload the code from this github repo <br>
2. Install raylib on your system. On windows this can be done with one of the commands below: <br>
`py -m pip install raylib`<br>
or<br>
`python -m pip install raylib`<br>

On linux:
`python3 -m pip install raylib`<br>

Make sure that you run python 3.12 or above

### usage

The software should be launched from a terminal to work properly.

Running just:

`python crane_simulator.py`

will spawn a GUI window for the user. The user can click on one of the boxes
by first pressing ESC and then selecting the container with the mouse. Pressing
ESC again will return you to camera control.

The utility has quite a few command line arguments

To see them all, type:

`python crane_simulator.py -h`

Here are some cool examples to get you started:

`python crane_simulator.py -w warehouse_config_file.csv` (this will create a different warehouse)

`python crane_simulator.py -i input_file.csv -S 100 -o output.csv` (this will go through inputs automatically, putting the results in output.csv)

`python crane_simulator.py -i input_file.csv --cli-only -o output.csv` (This will do the same as the command above but without the gui).

Some cool examples for linux users:

`python3 crane_simulator.py < input_file.csv --cli-only -o - | awk 'BEGIN { FS = ";" } ; { sum += 1 } END { if (NR > 0) print sum / NR / 60 "min" }'` (this will display the average cycle time for input_file.csv)

