class Labels:
    """
    Convenience class for label codes.
    """
    LEAF = 0
    WOOD = 1
    STEM = 2
    CONICAL = 11
    INVERSE_CONICAL = 12
    CYLINDRICAL = 13
    SPHERICAL = 14

    STR_DICT = {0: 'Leaf',
                1: 'Wood',
                2: 'Stem',
                11: 'Conical',
                12: 'Inverse Conincal',
                13: 'Cylindrical',
                14: 'Spherical'}

    @staticmethod
    def get_str(label):
        return Labels.STR_DICT[label]