pyculiarity
===========

A minimal and pure Python fork of @nicolasmiller's library [pyculiarity](https://github.com/nicolasmiller/pyculiarity).
That is a Python port of Twitter's AnomalyDetection R Package. The original source and examples are available
[here](https://github.com/twitter/AnomalyDetection).

This fork is focused on Python 3.10+ compatibility and no dependency on R. This is done using statsmodel's `tsa.seasonal_decompose`,
which differs in output from the Loess STL implementation used by the original pyculiarity library. The results are not
identical as a result, but are pretty close.

I've also stripped out some unused/unimplemented code to try to make this a little more readable/understandable. That
part is a work in progress. Part of that includes more intuitive handling of the timestamps, just have a 'timestamp'
column with unix timestamps in it.

Installation
------------

The original library is on pypi as pyculiarity, so to not clash with that, I've uploaded this as pyculiar. It will still
install the library as pyculiarity so this should function as a drop in replacement.

    pip install pyculiar

For development:

    pip install -e ".[dev]"

Usage
-----

As in Twitter's package, there are two top level functions, one for time-series data and one for simple vector
processing, detect_ts and detect_vec respectively. The first one expects a two-column Pandas DataFrame consisting of
timestamps and values. The second expects either a single-column DataFrame or a Series.

Here's an example of loading Twitter's example data (included in the tests directory) with Pandas and passing it to
Pyculiarity for processing.

    from pyculiarity import detect_ts
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib
    matplotlib.style.use('ggplot')

    __author__ = 'willmcginnis'

    if __name__ == '__main__':
        # first run the models
        twitter_example_data = pd.read_csv('../tests/raw_data.csv', usecols=['timestamp', 'count'])
        twitter_example_data['timestamp'] = pd.to_datetime(twitter_example_data['timestamp']).astype(int) // 10**9
        results = detect_ts(twitter_example_data, max_anoms=0.05, alpha=0.001, direction='both')

        # format the twitter data nicely
        twitter_example_data['timestamp'] = pd.to_datetime(twitter_example_data['timestamp'], unit='s')
        twitter_example_data.set_index('timestamp', drop=True)

        # make a nice plot
        f, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(twitter_example_data['timestamp'], twitter_example_data['count'], 'b')
        ax[0].plot(results['anoms'].index, results['anoms']['anoms'], 'ro')
        ax[0].set_title('Detected Anomalies')
        ax[1].set_xlabel('Time Stamp')
        ax[0].set_ylabel('Count')
        ax[1].plot(results['anoms'].index, results['anoms']['anoms'], 'b')
        ax[1].set_ylabel('Anomaly Magnitude')
        plt.show()

Which will give the plot:

![anomalies](https://github.com/wdm0006/pyculiarity/blob/master/examples/twitter_example.png)

Run the tests
-------------

    pytest

Copyright and License
---------------------

Changes Copyright 2016 Will McGinnis
Python port Copyright 2015 Nicolas Steven Miller
Original R source Copyright 2015 Twitter, Inc and other contributors

Licensed under the GPLv3
