# adclick-simulator-v2
Second generation of adclick simulator

## Updates

### 2019.04.29

Howto

- run competition.py to test run a competition
- tweak settings.spec to change the settings (e.g. policy mixes or seeds)

Highlights

- daily budgets are enforced
    - see ["budget"] entry in the data given to policies for the current remaining budget
    - when a policy is over the daily budget, its bids will become ineffective (set to 0)
    - at the beginning of every day, the budget will be replenished
- 40 location attributes are provided, along with 3 gender attributes and 7 age attributes.
- small auction fee is added in each cost-per-click    

## Get the Newest Code

A downloadable zip file is [here](https://github.com/donghun2018/adclick-simulator-v2/archive/master.zip). Also, you may clone this repository.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
See deployment for notes on how to deploy the project on a live system.

### Prerequisites

This software was tested with Python 3.6 running on Windows 10 and Ubuntu 16.04.

#### Python

Anaconda is an easy way to get a working python.
Get it [here](https://www.anaconda.com/download/).

This simulator is tested on a 64-bit python 3.6.8 as follows:
```
Python 3.6.8 |Anaconda, Inc.| (default, Feb 21 2019, 18:30:04) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

#### Packages

The packages required by the simulator are:

- numpy: used for pseudorandom number generator and many useful functions
- pandas: for dataframe and series manipulation

You may get these using conda

```
$ conda install numpy pandas
```

or using pip

```
$ pip install numpy pandas
```

### Installing

Get the source codes and try running

```
$ python test_sample_policies.py
```

This should output a lot of screen outputs, which is a running trace of the simulator with three policies over 24 time steps (equivalent to one simulated day)

### Other documents

- An introductory slideshow PDF file is available [here](https://github.com/donghun2018/adclick-simulator-v2/blob/release/20190424/documentation/20190415_ORF_418_adclick_game_intro.pdf)
- A "how-to" reference in coding a bidding policy is available [here](https://docs.google.com/document/d/1JJHlV3ORQG131_45ZCvQRToy4SdcIsElmKQGKohdM1M/edit?usp=sharing)
    - you may leave questions by commenting on this document 
    
## Contributors

### Roomsage Inc.

- Piotr Zioło (Lead)
- Jedrzej Kardach

### Castle Lab, Princeton Univ.

- Donghun Lee (Lead) 
- Andy Su

## Acknowledgments

- We appreciate Roomsage Inc. for its generous effort in development and support of this simulator.

## License

See the [LICENSE](https://github.com/donghun2018/adclick-simulator-v2/blob/master/LICENSE) file for details.