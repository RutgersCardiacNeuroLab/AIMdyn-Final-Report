# AIMdyn Final Report
### 1 The physiology model Python code

The physiology model solver and the associated data management, preparation
and processing procedures are packaged as a collection of Python scripts and
objects which can be either run separately or instrumented into a full processing
pipeline for the given subject data based on different requirements of the study.

#### 1.1 Data preparation and configuration

The initial data was given as a set of textual files representing timeseries for
different model variables, different subjects (designated by subject identification
numbers) and tasks (designated by task identification strings). Initial data
consolidation procedure was performed, collecting the variable data as well as
subject metadata (such as height, weight, sex, age and initial model state) and
storing it into a compressed format. The procedure produces a single .npz file
(which is a native numpy format for use in numerical computation using that
library) for each subject. These files are then further categorized by the task
ID. The file structure is used as the core input data storage for the entire model
architecture.

The configuration is setup as a pair of the configuration file containing paths
to data directories (the input directory and the results directory) and the path
to the external GOSUMD executable which need to be setup for running on
an individual machine, and the configuration object (as a Python script) which
loads in the available input data from the formatted data repository. The con-
figuration object can also load additional data such as the calculated initial state
from the results directory of a subject.

#### 1.2 Model setup

First step to setup the model is to designate the input data used for the run.
This is done by instantiating the configuration object and pointing the object to
the correct data to load based on the configuration JSON file and required task
and subject identification values. Next, it is necessary to create a physiology
model object and pass the configuration object as a constructor argument to
the model object. The model object serves as the interface between different
ways to run the computation, providing methods for storing the model analysis
results and auxiliary processing steps for the data.

The model object then has to initialize the model parameters by instantiating
the model parameter object which stores all parameters in a dictionary structure
providing named access to each of them. The model parameter object can either
be constructed prior to recalculation of the parameters with particular values set
by external actors or initialized from the defaults based on the subject metadata
where applicable. The state object is initialized in the same way so either by
providing an external state object or the state values can be filled in from the
input data (from the configuration object).

The thoracic and abdominal pressure for a subject is setup from the input
data (thoracic respiration) which initializes both timeseries for certain subject-
task configuration. This process initializes the necessary variables owned by
the model object. The data is then normalized based on the input systemic
pressure data. The final setup step sets up the initial systemic pressure and
time boundaries for the model run.

#### 1.3 Running the core model

Once the model is properly setup, it can be ran by calling the appropriate run
method. The run initializes the solver object which contains the code definition
of the delayed dynamical system with differential equations written out for each
state variable accessing the setup data. The step method for the solver takes
the previous state as an array which is unraveled during processing. An inter-
polation spline is calculated during the stepping process which creates anchors
dynamically to compute the delayed pressure state variable.

At the end of the model run the state object is updated with the resulting
state variables over the iteration period.

#### 1.4 Postprocessing and model analysis

During the initial run postprocessing step, the GOSUMD interface object is
initialized with the model object containing the parameters and results of the
model run. The object writes an initial GOSUMD parameter file for the batch
sensitivity run containing individual parameter distributions and distribution
bounds. Additionally, the object writes GOSUMD script files used by the exe-
cutable to setup the GOSUMD projects, analysis parameters and paths to data
files and model executables. Finally, the model is saved along with used pa-
rameter values, initial state and configuration input data to a pickle formatted
file.

A separate model run script is used as an executable model for GOSUMD
runs, both sensitivity and optimization. The difference in model setup for this
script is only in reading the parameters from a text file produced by a GOSUMD
iteration run, otherwise the model is setup the same way as for the initial run.

Additionally, at the end of the GOSUMD iteration run, objective function in-
dices are calculated based on the power spectrum density calculations of the
heart period and systemic pressure data computed by the model and the sub-
ject experimental data. The GOSUMD iteration run also saves the objective
function value and index values into a compatible text file which is used to
import the computed data into GOSUMD.

#### 1.5 Code structure

In the current version of the code the input files are saved in numpy .npz com-
pressed format where a single subject input file contains the following data:

* an array containing values of the measured blood pressure over time (BP)

* an array containing values of the measured RRI heart period for the sub-
ject over time (HP)

* an array containing values of measured diastolic arterial pressure over time
(DAP)
* an array containing values of measured systolic arterial pressure over time
(SAP)
* an array containing values of measured thoracic respiration volume over
time (TR)
* an array containing the initial state variable values for the subject
* subject id, weight, height, age and sex used to calculate subject parameter
values for the model
The code is split across Python scripts:
* physmod mat2npz.py - prepares the input data by reading all textual time-
series files and saves data into a single .npz file with the same name as

the subject folder from where the data was read; a text file with subject
heights and weights also needs to be located in the data folder;

* physmod batch run.py - used to run the model in a multiprocessing, batch
manner, reads the data for the subject from the input file (providing the
path to the input file and setting up the time discretization for the run
is necessary), uses the paths setup in batch config.json to run external
software;
* physmod config.py - provides an interface for the subject’s input data
* physmod equations.py - defines the model equations along with the step
function and additional utility functions
* physmod gosumd run.py - sets up and runs the model, used as a model
script file for running the model through GOSUMD
* physmod gosumd.py - utility script for creating GOSUMD script files for
a single subject
* physmod initial run.py - sets up and runs the model using the input initial
state, produces a revised initial state and GOSUMD script files for a
specific subject
* physmod model.py - core container script, provides access to data pro-
cessing methods as well as the entry point for running the model
* physmod params.py - container for model parameters, provides utility
functions for processing the input parameters
* physmod sens postprocess.py - utility script for processing the sensitivity
results
* physmod state.py - container for the model state, gets updated with the
resulting values after the model is run
* physmod vis.py - structure for loading the model data and providing func-
tions for useful visualizations of the model contents

The zip archive with the Python code is supplied to the Rutgers University
as an attachment to the email with the final report.
