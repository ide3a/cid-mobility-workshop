import subprocess
import sys
import os
import re
import json
from xml.etree import ElementTree

import pandas as pd


try:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import sumolib
except ImportError:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


class EventLog:

    def __init__(self, df_vehicle_updates, df_v2x_message_transmission, df_v2x_message_reception, df_cell_configuration,
                 df_adhoc_configuration, df_vehicle_registration, df_traffic_log, df_navigation_log):
        """Simulation result.

        Args:
            df_vehicle_updates: VEHICLE_UPDATES events
            df_v2x_message_transmission: V2X_MESSAGE_TRANSMISSION events
            df_v2x_message_reception: V2X_MESSAGE_RECEPTION events
            df_cell_configuration: CELL_CONFIGURATION events
            df_adhoc_configuration: ADHOC_CONFIGURATION events
            df_vehicle_registration: VEHICLE_REGISTRATION events
            df_traffic_log: Contents of `Traffic.log` file
            df_navigation_log: Contents of `Navigation.log` file
        """
        self.df_vehicle_updates = df_vehicle_updates
        self.df_v2x_message_transmission = df_v2x_message_transmission
        self.df_v2x_message_reception = df_v2x_message_reception
        self.df_cell_configuration = df_cell_configuration
        self.df_adhoc_configuration = df_adhoc_configuration
        self.df_vehicle_registration = df_vehicle_registration
        self.df_traffic_log = df_traffic_log
        self.df_navigation_log = df_navigation_log

        # Calculate how many vehicles already got messages from the V2X network
        self.df_navigation_log['Vehicles_List'] = ''
        self.df_navigation_log['Vehicles_List'][self.df_navigation_log[12].str.contains('veh_') == True] = \
            self.df_navigation_log[12].str.split('.').str[0]
        self.df_navigation_log = self.df_navigation_log[self.df_navigation_log['Vehicles_List'] != '']

        # Change column names and group by network types of 'df_v2x_message_transmission'
        self.df_v2x_message_transmission.rename(
            columns={'SourceName': 'Name', 'Type': 'TransmissionType', 'MessageId': 'TransmissionMessageId'},
            inplace=True)

        # Change column names and group by network types of 'df_v2x_message_reception'
        self.df_v2x_message_reception.rename(
            columns={'ReceiverName': 'Name', 'Type': 'ReceiverType', 'MessageId': 'ReceiverMessageId'}, inplace=True)

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

    def run_simulation(self, visualize=True, sim_speed=None) -> None:
        """Run the selected simulation and record logs"""
        extension = './mosaic.sh' if os.name == 'posix' else 'mosaic.bat'
        shell = False if os.name == 'posix' else True
        command = [extension, '-s', self.sim_name]
        if visualize:
            command.append('-v')
        if sim_speed is not None:
            command.append('-b')
            command.append(str(sim_speed))
        print(f"Running: {' '.join(command)}")

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
        print(f'The path for the logs: {log_path}')
        try:
            dirs = sorted([f.name for f in os.scandir(log_path) if f.is_dir()],
                          reverse=True)
        except FileNotFoundError:
            print("Warning: Could not load any existing simulation results.")
            return

        if not len(dirs)==0:
            print(f"Log List size is {len(dirs)}")
            self.sim_select = os.path.join(log_path, dirs[idx])
            latest = "latest " if idx == 0 else ""
            print(f"Loading {latest}simulation result '{dirs[idx]}'")
        else:
            print("Warning: Could not load any existing simulation results.")

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

    def _get_log_file(self, log_name) -> pd.DataFrame:
        log_data  = open(os.path.join(self.sim_select + log_name), 'r')
        result={}
        i=0
        for line in log_data:
            columns = line.split(' ')
            result[i] = columns
            i+=1    
        j=json.dumps(result)
        df= pd.read_json(j,orient='index') 
        return df

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
                           names=col_names,on_bad_lines='skip', engine ='python')

    def _get_output_config(self):
        xml_path = os.path.join(self.mosaic_path,
                                'scenarios',
                                self.sim_name,
                                'output',
                                'output_config.xml')

        tree = ElementTree.parse(xml_path)
        return tree.getroot()

    def import_data(self) -> EventLog:
        """Import and parse the data from the output files"""
        df_navigation_log = self._get_log_file('/Navigation.log')
        df_traffic_log = self._get_log_file('/Traffic.log')

        # filter dataframe to only include rows with 'VEHICLE_UPDATES'
        df_vehicle_updates = self.filter_df(Event='VEHICLE_UPDATES',
                            select=['VehicleEmissionsCurrentEmissionsCo2',
                                    'VehicleEmissionsAllEmissionsCo2',
                                    'Name',
                                    'RouteId',
                                    'RoadPositionLaneIndex'])
        # filter dataframe to only include rows with 'V2X_MESSAGE_TRANSMISSION'   
        df_v2x_message_transmission = self.filter_df(Event='V2X_MESSAGE_TRANSMISSION',
                                                     select=['Type', 'MessageId','SourceName','MessageRoutingDestinationType'])
        # filter dataframe to only include rows with 'V2X_MESSAGE_RECEPTION'   
        df_v2x_message_reception = self.filter_df(Event='V2X_MESSAGE_RECEPTION',
                                                  select=['Type', 'MessageId','ReceiverName','ReceiverInformationReceiveSignalStrength'])
        # filter dataframe to only include rows with 'CELL_CONFIGURATION'   
        df_cell_configuration = self.filter_df(Event='CELL_CONFIGURATION',
                                               select=['ConfigurationNodeId'])
        # filter dataframe to only include rows with 'ADHOC_CONFIGURATION'   
        df_adhoc_configuration = self.filter_df(Event='ADHOC_CONFIGURATION',
                                                select=['ConfigurationNodeId'])
        # filter dataframe to only include rows with 'VEHICLE_REGISTRATION'   
        df_vehicle_registration = self.filter_df(Event='VEHICLE_REGISTRATION',
                                                 select=['MappingGroup', 'MappingName', 'MappingVehicleTypeVehicleClass']
                                                 ).rename(columns={'MappingName': 'Name'})

        return EventLog(df_vehicle_updates, df_v2x_message_transmission, df_v2x_message_reception, df_cell_configuration,
                        df_adhoc_configuration, df_vehicle_registration, df_traffic_log, df_navigation_log)
