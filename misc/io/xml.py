import re
from io import StringIO
from xml.etree import cElementTree

import numpy as np


def control_in(control_file):
    """Parses a control input file.

    Args:
        control_file (str): The path to the control file.
    """
    pass


class Box(object):
    """Represents the simulation box dimensions and tilt factors.

    Attributes:
        xy (float): Tilt factor xy.
        xz (float): Tilt factor xz.
        yz (float): Tilt factor yz.
    """
    def __init__(self):
        """Initializes a Box object with zero tilt factors."""
        self.xy = 0
        self.xz = 0
        self.yz = 0
        return

    def update(self, dic):
        """Updates the box attributes from a dictionary.

        Args:
            dic (dict): Dictionary containing box attributes.
        """
        self.__dict__.update(dic)


class XmlParser(object):
    """Parses XML files containing simulation data.

    This class reads an XML file and extracts specific data elements, optionally filtering
    by a list of needed tags. It handles specific structures like simulation boxes,
    reaction lists, and templates, while parsing other data into numpy arrays.

    Attributes:
        box (Box): The simulation box object.
        data (dict): A dictionary storing parsed data arrays and structures (e.g., 'reaction', 'template').
    """
    def __init__(self, filename, needed=None):
        """Initializes the XmlParser.



        Parsing logic:
        1. Reads root attributes and sets them as instance attributes (e.g., step, time).
        2. Iterates through child elements.
        3. 'box' tags update the self.box object.
        4. Other tags are filtered against the `needed` list if provided.
        5. 'reaction' tags are parsed into lists of reaction steps.
        6. 'template' tags are evaluated as Python dictionaries.
        7. All other tags are parsed as numpy arrays from text content.

        Args:
            filename (str): Path to the XML file to parse.
            needed (list, optional): A list of XML tag names to extract. If provided,
                tags not in this list (except 'box') are skipped. Defaults to None.
        """
        tree = cElementTree.ElementTree(file=filename)
        root = tree.getroot()
        self.box = Box()
        self.data = {}
        needed = [] if needed is None else needed
        for key in root[0].attrib:
            self.__dict__[key] = int(root[0].attrib[key])
        for element in root[0]:
            if element.tag == 'box':
                self.box.update(element.attrib)
                continue
            if (len(needed) > 0) and (element.tag not in needed):
                continue
            if element.tag == 'reaction':
                self.data['reaction'] = []
                reaction_list = element.text.strip().split('\n')
                while '' in reaction_list:
                    reaction_list.remove('')
                for l in reaction_list:
                    r = re.split(r'\s+', l)
                    while '' in r:
                        r.remove('')
                    r[1:] = [int(_) for _ in r[1:]]
                    self.data['reaction'].append(r)
                continue
            if element.tag == 'template':
                self.data['template'] = eval('{%s}' % element.text)
                continue
            if len(element.text.strip()) > 0:
                self.data[element.tag] = np.genfromtxt(StringIO(element.text), dtype=None, encoding=None)