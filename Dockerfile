FROM waggle/plugin-base:1.1.1-ml
COPY requirements.txt /app/


RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt
COPY . /app/

ADD https://web.lcrc.anl.gov/public/waggle/models/jszumny/attempts/3features/vgg16.pt /app/vgg16.pt

WORKDIR /app
ENTRYPOINT ["python3", "/app/main.py"]
