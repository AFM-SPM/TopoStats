# Gwyddion with Python base image
FROM afmspm/gwyddion-python

# Topostats requirements
RUN apt-get update && apt-get install -y \
    python-matplotlib \
    python-pandas \
    python-pip \
    python-seaborn \
    python-skimage