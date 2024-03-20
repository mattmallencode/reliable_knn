PYTHON=python3

VENV_DIR=venv

PACKAGE_PATH=$(VENV_DIR)/lib/python3.10/site-packages/imblearn/under_sampling/_prototype_selection/_edited_nearest_neighbours.py

CUSTOM_FILE=_edited_nearest_neighbours.py

all: setup replace_file

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r requirements.txt
	mkdir results

replace_file:
	cp $(CUSTOM_FILE) $(PACKAGE_PATH)

clean:
	rm -rf $(VENV_DIR)
