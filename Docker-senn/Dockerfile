FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel

RUN apt-get update &&\
    apt-get install -y python-tk

RUN pip install cython &&\
    pip install numpy==1.14.0 &&\
    pip install scipy==1.0.0 &&\
    pip install matplotlib==2.1.0 &&\
    pip install scikit-learn==0.19.1  scikit-image==0.13.1 scikit-optimize==0.5.2 &&\
    pip install tqdm==4.19.4 &&\
    pip install attrdict==2.0.0 &&\
    pip install pandas==0.20.3 &&\
    pip install lime 

RUN git clone https://github.com/dmelis/SENN.git &&\
    cd SENN &&\
    pip install ./

RUN install -d -m 0755 -o root -g users /workspace/ExpO

VOLUME /workspace/ExpO

