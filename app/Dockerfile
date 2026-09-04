# copy the the nicegui base image
FROM zauberzeug/nicegui
WORKDIR /app

RUN apt-get update && apt-get install -y \
libgl1 \
libglib2.0-0 \
libsm6 \
libxext6 \
libxrender1 \
libx11-6 \
libxcb1 \
&& rm -rf /var/lib/apt/lists/*

# install dependencies
COPY requirements.txt .
RUN uv pip install -r requirements.txt

# copy source code and set up port
COPY . .
EXPOSE 8080

CMD ["python", "main.py"]
