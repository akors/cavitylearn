# Dockerfile for cavitylearn package and scripts
# Alexander Korsunsky, 2021
#
# Run this from the top-level cavitylearn directory

FROM tensorflow/tensorflow:1.15.5-gpu-py3
LABEL org.opencontainers.image.authors="Alexander Korsunsky"
LABEL org.opencontainers.image.title="cavitylearn"
LABEL org.opencontainers.image.description="Deep learning on protein cavities"

ADD py /src/cavitylearn/py
ADD ipy /src/cavitylearn/ipy

RUN pip3 install --no-cache-dir -r /src/cavitylearn/py/requirements.txt notebook
ENV PATH="${PATH}:/src/cavitylearn/py" PYTHONPATH="${PYTHONPATH}:/src/cavitylearn/py"

EXPOSE 6006 8888
