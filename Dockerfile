# Gwyddion with Python base image
FROM afmspm/gwyddion-python

# Topostats requirements
RUN apt-get --allow-releaseinfo-change update && apt-get install -y \
    python-matplotlib \
    python-pandas \
    python-pip \
    python-seaborn \
    python-skimage

# Dummy display
ENV DISPLAY=":1"
COPY debugutils/dummy_display.sh /opt/dummy_display.sh
RUN chmod a+x /opt/dummy_display.sh
CMD /opt/dummy_display.sh
