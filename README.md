# sweng-capstone-project
Repository for PSU SWENG 894

Prerequisites:
Python 3.13
git CLI

To run the server:

Open a command line

Run the following: 

git clone https://github.com/danmcgill9/sweng-capstone-project.git

cd sweng-capstone-project

python -m venv venv

  Linux Users:  source venv/bin/activate
  
  Windows Users: ./venv/Scripts/activate

pip install -r requirements.txt

cd proj

python manage.py runserver

Open a browser and navigate to http://localhost:8000

## Running Celery and RabbitMQ (Task Queue Setup)

This project uses **Celery** for background task processing and **RabbitMQ** as the message broker.

---

### Prerequisites

- Python virtual environment activated
- RabbitMQ and ERLANG installed and available in your system path
- Project dependencies installed (`pip install -r requirements`)
- RabbitMQ and Celery must be run in separate terminals for runserver (see instructions below)

---

### Start RabbitMQ

```bash
# On Linux
rabbitmq-server

# On Windows Powershell with Admin (if installed manually)
rabbitmq-server start
```
---

### Start Celery
```bash
# On Linux, dir where manage.py is located (proj)
celery -A modelworx worker --loglevel=info --pool=solo

# For development, only one thread will be running, tasks will be queued in the one thread.
```

---

### Start Application 
```bash
# On Linux
python manage.py makemigrations  # to capture latest models changes
python manage.py migrate
python manage.py runserver
```
