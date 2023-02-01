FROM python:3.10

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get autoremove 
RUN apt-get autoclean

RUN mkdir /PostNeuronia
COPY . /PostNeuronia/
WORKDIR /PostNeuronia

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "src/bot.py" ]