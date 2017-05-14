#!/usr/bin/env python3
import sys
import re


def get_data(file_name):
    with open(file_name, 'r') as fin:
        whole_data = fin.read()
        data_array = whole_data.split(',')
        for index, d in enumerate(data_array):
            data_array[index] = d.strip()
            print(data_array[index])
        p = re.compile(r'D-(?P<day>\d+)/(?P<month>\d+)/(?P<year>\d+)')
        fa = file_name.split('.')
        with open(fa[0] + "_out." + fa[1], 'w') as fout:
            ans = []
            for element in data_array:
                if "to" in element:
                    print(element)
                    t = element.split("to")
                    start = t[0]
                    end = t[1]
                    matches = p.search(start)
                    start_day = int(matches['day'])
                    month = int(matches['month'])
                    year = int(matches['year'])
                    end_day = int(p.search(end)['day'])

                    for index in range(start_day, end_day + 1):
                        print(index)
                        ans.append("D-%d/%d/%d" % (index, month, year))

                else:
                    element = element.strip()
                    ans.append(element)
            fout.write(str.join(',', ans))


if __name__ == '__main__':
    print(sys.argv[1])
    data = get_data(sys.argv[1])
