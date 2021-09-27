FROM kaggle/python-gpu-build
RUN mkdir /clrp
WORKDIR /clrp
ENV NVIDIA_VISIBLE_DEVICES all
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
EXPOSE 8888
RUN pip install -U pip && \
    pip install transformers