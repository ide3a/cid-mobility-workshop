import subprocess
import sys
import os
import re
import json
from lxml import etree

import numpy as np
import pandas as pd
import glom
import pyproj

from typing import Any


try:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import sumolib
except ImportError:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


class Mosaic:
    """Initialize the Mosaic toolbox

    Parameters
    ----------
    mosaic_path : str
        path to Eclipse MOSAIC
    sim_name : str
        Simulation name
    """
    def __init__(self,
                 mosaic_path: str,
                 sim_name: str = 'Barnim') -> None:
        self.sim_name = sim_name
        self.mosaic_path = mosaic_path
        self.set_simulation_result()

    def run_simulation(self, visualize=True) -> None:
        """Run the selected simulation and record logs
        """
        extension = './mosaic.sh' if os.name == 'posix' else 'mosaic.bat'
        shell = False if os.name == 'posix' else True
        command = [extension, '-s', self.sim_name]
        if visualize:
            command.append('-v')
        print("Running: " + " ".join(command))
        output = subprocess.check_output(command,
                                         stderr=subprocess.STDOUT,
                                         cwd=self.mosaic_path,
                                         shell=shell)
        print(output.decode('ascii'))
        self.set_simulation_result()

    def set_simulation_result(self, idx: int = 0):
        """Utility function to select the simulation and generate DataFrames
        IMPORTANT: Always run this function first after run_simulation() and
        before any other getter/setter and diverse functions!

        Parameters
        ----------
        idx : int, optional
            index of the log, 0 is the most recent result from
            the simulation, 1 is the second most recent, by default 0
        """
        log_path = os.path.join(self.mosaic_path, 'logs')
        try:
            dirs = sorted([f.name for f in os.scandir(log_path) if f.is_dir()],
                          reverse=True)
        except FileNotFoundError:
            print("Warning: Could not load any existing simulation results.")
            return

        self.sim_select = os.path.join(log_path, dirs[idx])
        latest = "latest " if idx == 0 else ""
        print(f"Loading {latest}simulation result '{dirs[idx]}'")

        output_root = self._get_output_config()

        id2fields = dict()
        for elem in output_root[0][3]:
            k = elem[0][0].text.replace('"', '')
            v = [re.sub(r"Updated:", '', i.text) for i in elem[0]]
            v = [re.sub(r"\.", '', i) for i in v]
            v[0] = 'Event'
            id2fields[k] = v
        self.id2fields = id2fields

    def filter_df(self, **kwargs) -> pd.DataFrame:
        """Filter DataFrame using the event name, application name and
        fields

        Parameters
        ----------
        **kwargs : field=value
            Filter by field-value pair

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        assert 'Event' in kwargs, 'Must specify an event name'
        assert 'select' in kwargs, 'Either "all" or list of str'
        if kwargs['select'] == 'all':
            selected = 'all'
        else:
            assert isinstance(kwargs['select'], list)
            selected = kwargs['select']

        col_names = self.id2fields[kwargs['Event']]
        output_df = self._get_output_csv(col_names)

        # Cleanup
        del kwargs['select']

        # Boolean filters
        for k, v in kwargs.items():
            is_df_bool = output_df[k] == v
            output_df = output_df[is_df_bool]

        if selected != 'all':
            list_diff = list(set(col_names)
                             - set(['Event', 'Time'])
                             - set(selected))

            filtered_df = output_df.drop(list_diff, axis=1)
        else:
            filtered_df = output_df

        return filtered_df

    def _get_output_csv(self, col_names) -> pd.DataFrame:
        """Getter function for the output.csv file, which holds the log data of
        the indexed simulation.

        Returns
        -------
        pd.DataFrame
            DataFrame of output.csv
        """
        return pd.read_csv(os.path.join(self.sim_select + '/output.csv'),
                           sep=';',
                           header=None,
                           names=col_names)

    def _get_output_config(self):
        xml_path = os.path.join(self.mosaic_path,
                                'scenarios',
                                self.sim_name,
                                'output',
                                'output_config.xml')

        tree = etree.parse(xml_path)
        return tree.getroot()

    def df2np(self, df: pd.DataFrame) -> np.array:
        """Convert DataFrame to Numpy array

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame

        Returns
        -------
        np.array
            np.asfarray(df1)
        """
        return np.asfarray(df)

    def retrieve_federate(self, federate: str, idx=None) -> str:
        """Retrieves the selected federate and initializes object attribute for
        further actions

        Parameters
        ----------
        federate : str
            Federate name, can be 'scenario' or any of the other federates,
            i.e. 'application', 'sns', 'cell', etc.
        idx : int, optional
            Index of the federate within a federate folder, if idx=None, no
            federate is selected and available options are displayed

        Returns
        -------
        str
            Full path of the federate json
        """

        if federate == 'scenario':
            path_to_fedjson = os.path.join(self.mosaic_path,
                                           'scenarios',
                                           self.sim_name,
                                           'scenario_config.json')
            self.fed_path = path_to_fedjson
            foo = open(path_to_fedjson)
            self.fed_name = 'scenario_config.json'
            self.current_fed_setting = json.load(foo)
            return path_to_fedjson

        else:
            path_to_json = os.path.join(self.mosaic_path,
                                        'scenarios',
                                        self.sim_name,
                                        federate)

            json_files = sorted([pos_json for pos_json
                                 in os.listdir(path_to_json)
                                 if pos_json.endswith('.json')])

            if idx is None:
                print(json_files)
                return json_files
            else:
                assert isinstance(idx, int)
                path_to_fedjson = os.path.join(path_to_json, json_files[idx])
                self.fed_path = path_to_fedjson
                foo = open(path_to_fedjson)
                self.fed_name = json_files[idx]
                self.current_fed_setting = json.load(foo)
                return path_to_fedjson

    def set_federate_value(self, tree: str, value: str) -> None:
        """Sets the selected federate value in the tree and dumps the new
        federate configuration into JSON

        Parameters
        ----------
        tree : str
            dict tree to navigate to the federate value to be changed, i.e.
            'globalNetwork.uplink.delay.delay'
        value : str
            value to set, IMPORTANT: make sure you follow the format of the
            federate value
        """
        glom.assign(self.current_fed_setting, tree, val=value)
        print('Federate value of {} set to {}'.format(
            tree, value
        ))
        with open(self.fed_path, 'w') as f:
            json.dump(self.current_fed_setting, f, indent=4)
        pass

    def get_federate_value(self, tree: str) -> Any:
        """Gets the selected federate value in the tree

        Parameters
        ----------
        tree : str
            dict tree to navigate to the federate value to be changed, i.e.
            'globalNetwork.uplink.delay.delay'

        Returns
        -------
        Any
            federate value
        """

        return glom.glom(self.current_fed_setting, tree)

    @property
    def pprint_curr_fed(self):
        """Pretty print current federate configuration
        """
        print('Current federate: {}'.format(self.fed_name))
        print(json.dumps(self.current_fed_setting, indent=4, sort_keys=True))

    @property
    def get_federates(self):
        """Print available federate configurations
        """
        path_to_settings = os.path.join(self.mosaic_path,
                                        'scenarios',
                                        self.sim_name)
        print('Available federates: {}'.format(sorted(
            [f.name for f in os.scandir(path_to_settings)])))

    @property
    def get_df_apps(self) -> list:
        """Getter DataFrame Applications

        Returns
        -------
        list
            DataFrame Applications
        """
        app_dir = os.path.join(self.sim_select, 'apps')
        return sorted([f.name for f in os.scandir(app_dir) if f.is_dir()],
                      reverse=True)

    @property
    def get_df_events(self) -> list:
        """Getter DataFrame Events

        Returns
        -------
        list
            DataFrame Events
        """
        return list(self.id2fields.keys())

    def get_df_labels(self, event: str) -> list:
        """Getter DataFrame Fields

        Parameters
        ----------
        event : str
            Event type

        Returns
        -------
        list
            DataFrame Fields
        """
        return self.id2fields[event]

    @property
    def get_output_df(self) -> pd.DataFrame:
        """Getter output DataFrame

        Returns
        -------
        pd.DataFrame
            output DataFrame
        """
        return self.output_df

    """
    def plotter(self) -> None:
        path_net = os.path.join(self.mosaic_path,
                                'scenarios',
                                self.sim_name,
                                'sumo',
                                self.sim_name + '.net.xml')

        net = sumolib.net.readNet(path_net)
        x_off, y_off = net.getLocationOffset()

        p = pyproj.Proj(proj='utm',
                        zone=33,
                        ellps='WGS84',
                        preserve_units=False)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        shapes = [elem.getShape() for elem in net._edges]

        shapes_geo = []

        for shape in shapes:
            foo = [(p(el[0] - x_off, el[1] - y_off,
                      inverse=True)) for el in shape]
            shapes_geo.append(foo)

        line_segments = LineCollection(shapes_geo, colors='k', alpha=0.5)
        ax.add_collection(line_segments)
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
        ax.autoscale_view(True, True, True)
        ax.set_ylim([52.60, 52.653])
        ax.set_xlim([13.51, 13.57])

        # Add RSUs
        rsu_0 = self.filter_df(Event='RSU_REGISTRATION',
                               MappingName='rsu_0',
                               select='all')

        ax.scatter(rsu_0.MappingPositionLongitude.astype(float),
                   rsu_0.MappingPositionLatitude.astype(float),
                   c='g', linewidths=20, marker="1", label='Road Side Unit')

        # Get road conditions
        self.retrieve_federate('environment', idx=0)
        spec = ('events', ['location.area.a'])
        envpts_a = sorted(list(self.get_federate_value(spec)[0].values()))
        spec = ('events', ['location.area.b'])
        envpts_b = sorted(list(self.get_federate_value(spec)[0].values()))

        lon = (envpts_a[0], envpts_b[0])
        lat = (envpts_a[1], envpts_b[1])

        rect = Rectangle((min(lon), min(lat)),
                         max(lon)-min(lon),
                         max(lat) - min(lat),
                         linewidth=5, edgecolor='c',
                         facecolor='none', label='Hazardous Road')

        ax.add_patch(rect)
    """

    def eval_simulation(self) -> list:
        """Evaluate simulation

        Returns
        -------
        list
            [num vehicles standard route,
            num vehicles alternate route,
            co2 emissions]

        """
        c = [13.54995, 52.63254, 0.01]  # x, y, r

        df = self.filter_df(Event='VEHICLE_UPDATES',
                            select=['PositionLongitude',
                                    'PositionLatitude',
                                    'Name'])

        veh_total = len(set(df.Name))

        gps_obs = np.concatenate(
            (np.asfarray(df.PositionLongitude).reshape(-1, 1),
             np.asfarray(df.PositionLatitude).reshape(-1, 1)), axis=1)

        veh_in_circle = list()

        for idx, val in enumerate(gps_obs):
            foo = self._in_circle(val, c)
            if foo is True:
                veh_in_circle.append(df.Name.iloc[idx])
            else:
                pass

        set_v2r = set(veh_in_circle)
        veh2alt = len(set_v2r)
        veh2std = veh_total - veh2alt

        # CO2 Emissions
        df = self.filter_df(Event='VEHICLE_UPDATES',
                            select=['Name', 'VehicleEmissionsAllEmissionsCo2'])
        co2_per_car = df[["Name", "VehicleEmissionsAllEmissionsCo2"]].groupby("Name").max()
        co2_mean = co2_per_car.mean()[0] / 1000

        print("{} vehicles took the standard route".format(veh2std))
        print("{} vehicles took the alternate route".format(veh2alt))
        print("On average a vehicle released {:.2f} g CO2".format(co2_mean))

        return veh2std, veh2alt, co2_mean

    def _in_circle(self, p, c):
        xp, yp = p[0], p[1]
        xc, yc, r = c

        d = np.sqrt((xp-xc)**2 + (yp-yc)**2)

        if r > d:  # Point is in circle
            return True
        else:
            return False

        '''
        df = self.filter_df(Event='VEHICLE_UPDATES',
                            select=['PositionLongitude',
                                    'PositionLatitude'])

        gps_obs = np.concatenate(
            (np.asfarray(df.PositionLongitude).reshape(-1, 1),
             np.asfarray(df.PositionLatitude).reshape(-1, 1)), axis=1)

        _dens_labels = np.array([self._classify(obs) for obs in gps_obs])

        r0 = gps_obs[_dens_labels == -1.].T
        r1 = gps_obs[_dens_labels == 1.].T

        # weights of road
        total_pts = gps_obs.size
        w0 = r0.size / total_pts
        w1 = r1.size / total_pts

        ax.scatter(r0[0], r0[1], c='r',
                   linewidths=1,
                   label='Density = {}'.format(w0),
                   alpha=w0)
        ax.scatter(r1[0], r1[1], c='b',
                   alpha=w1,
                   linewidths=1,
                   label='Density = {}'.format(w1))
        '''
        # plt.legend(loc='upper left')

    def _classify(self, gps_coord):
        A = [13.5359800, 52.6128399]
        B = [13.567001, 52.644249]

        if gps_coord[0] < A[0]:
            return 0.
        elif gps_coord[1] > B[1]:
            return 0.
        else:
            position = np.sign((B[0] - A[0])
                               * (gps_coord[1] - A[1])
                               - (B[1] - A[1])
                               * (gps_coord[0] - A[0]))

        return position
