import sys
from math import cos, sin, pi, floor, ceil

class AccuracyGraph():
    def __init__(self, max_values=50, height=20, plt_config={}):
        self.__values = []
        self.__last_message = 'Accuracy graph'
        self.__last_output = ''
        self.__max_values = max_values
        self.__height = height
        self.__plt_config = plt_config

    def refresh(self, msg):
        self.__print_with_message(msg)

    def update_accuracy(self, accuracy, msg=None):
        if msg is None:
            msg = self.__last_message

        self.__last_message = msg

        # Append and limit values
        self.__values.append(accuracy)
        self.__values = self.__values[-self.__max_values:]
        self.__print_with_message(self.__last_message)

    def __clear_chars(self):
        line_count = self.__last_output.count('\n') + 1

        return ''.join(['\033[F' for _ in range(line_count)])

    def __print_with_message(self, msg):
        output = self.__clear_chars()

        if len(self.__values) > 1:
            output += self.__plot_lines(self.__values, self.__plt_config)

        output += '\n' + msg
        self.__last_output = output
        print(output)

    def __plot_lines(self, series, cfg={}):
        """asciichart-adapted code"""

        minimum = cfg['minimum'] if 'minimum' in cfg else min(series)
        maximum = cfg['maximum'] if 'maximum' in cfg else max(series)

        interval = abs(float(maximum) - float(minimum))
        interval = 1 if interval == 0 else interval

        offset = cfg['offset'] if 'offset' in cfg else 3
        padding = cfg['padding'] if 'padding' in cfg else '           '
        height = self.__height
        ratio = height / interval
        min2 = floor(float(minimum) * ratio)
        max2 = ceil(float(maximum) * ratio)
        column_width = cfg['column_width'] if 'column_width' in cfg else 80

        intmin2 = int(min2)
        intmax2 = int(max2)

        rows = abs(intmax2 - intmin2)
        width = len(series) + offset
        placeholder = cfg['format'] if 'format' in cfg else '{:8.2f} '

        result = [[' '] * width for i in range(rows + 1)]

        # axis and labels
        for y in range(intmin2, intmax2 + 1):
            label = placeholder.format(float(maximum) - ((y - intmin2) * interval / rows))
            result[y - intmin2][max(offset - len(label), 0)] = label
            result[y - intmin2][offset - 1] = '┼' if y == 0 else '┤'

        y0 = int(series[0] * ratio - min2)
        result[rows - y0][offset - 1] = '┼' # first value

        for x in range(0, len(series) - 1): # plot the line
            y0 = int(round(series[x + 0] * ratio) - intmin2)
            y1 = int(round(series[x + 1] * ratio) - intmin2)
            if y0 == y1:
                result[rows - y0][x + offset] = '─'
            else:
                result[rows - y1][x + offset] = '╰' if y0 > y1 else '╭'
                result[rows - y0][x + offset] = '╮' if y0 > y1 else '╯'
                start = min(y0, y1) + 1
                end = max(y0, y1)
                for y in range(start, end):
                    result[rows - y][x + offset] = '│'

        return '\n'.join([''.join(row).ljust(column_width) for row in result])
