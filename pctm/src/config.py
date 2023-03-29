# PointCloud_Tree_Modelling by Amsterdam Intelligence, GPL-3.0 license

import os
import logging

logger = logging.getLogger()


class Paths:
    """
    Convenience class for executable paths.
    """

    ADTREE = '../../bin/AdTree' # path of AdTree executable

    @staticmethod
    def get_adtree():
        if not os.path.isfile(Paths.ADTREE):
            logger.warning('WARNING: No file exists at AdTree path. Please set correct path in `./pctm/src/config.py`')

        return Paths.ADTREE

