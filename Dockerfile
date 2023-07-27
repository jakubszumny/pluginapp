FROM waggle/plugin-base:1.1.1-ml


ADD https://web.lcrc.anl.gov/public/waggle/models/jszumny/attempts/3features/vgg16.pt /app/vgg16.pt
COPY requirements.txt /app/


RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt
COPY . /app/

WORKDIR /app
ENTRYPOINT ["python3", "/app/app.py"]
