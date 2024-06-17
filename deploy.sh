#!/bin/bash

# Enter the wwwroot directory
cd /home/site/wwwroot

# Remove old installation if it exists
if [ -d "env" ]; then
    rm -rf env
fi

# Create and activate a virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run migrations if necessary
# python manage.py migrate

# Collect static files if necessary
# python manage.py collectstatic --noinput

# Deactivate the virtual environment
deactivate
